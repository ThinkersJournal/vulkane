// SPDX-License-Identifier: MIT OR Apache-2.0
//! `Backend::lower_kernel`: the kernel layer (`unpopped::kernel::KernelFn`) lowered to a SPIR-V
//! compute module. ROADMAP rows 8a (workgroup memory) and 8b (barriers and fences), first slice.
//!
//! ## What v1 lowers
//!
//! - **Integer kernels only**: `compute` is `I32` or `U32`, and every buffer, shared array and
//!   local has that element type. A float kernel is a typed decline (the thread ids are integers,
//!   and a float kernel needs casts this slice does not spell).
//! - **Parameters**: `Buffer`, one storage buffer per parameter at set 0, binding = the
//!   parameter's index (a `Read` buffer is decorated `NonWritable`), and the extent scalar above.
//!   Other `Scalar` parameters, `Struct` parameters, multi-coordinate places and peer/multicast
//!   places are declined.
//! - **Launch**: the kernel must name a [`LaunchShape`](unpopped::exec::LaunchShape); its `block`
//!   is the `LocalSize`, `BlockDim`/`GridDim` are constants from it. `grid_dims` that depend on a
//!   parameter are declined.
//! - **Workgroup memory** (row 8a): `MemorySpace::Shared` arrays become `OpVariable Workgroup` of
//!   array type. A dynamic extent (`Extent::Param`) names a scalar parameter of the compute dtype,
//!   which is lowered as an `OpSpecConstant` (default 1) whose `SpecId` is **the parameter's index**;
//!   the array length is that constant and the parameter's local takes its value at entry. **The
//!   extent is therefore fixed per pipeline, not per dispatch**: the host sets it when it creates the
//!   pipeline. A scalar parameter that does not size a shared array is a typed decline.
//! - **Statements**: `Let`, `Load`, `Assign`, `Store`, `Barrier`, `Fence` and the structured control
//!   flow of row 8g (`If`, `For`, `While`, `Break`, `Continue`, `Return`; see the `flow` module). Atomics,
//!   collectives and everything else is declined naming the statement.
//!
//! ## Barriers and fences (row 8b)
//!
//! `Barrier { Block, Full }` is `OpControlBarrier Workgroup Workgroup AcquireRelease|WorkgroupMemory|UniformMemory`.
//! **Deviation from the ROADMAP text, deliberate:** the semantics also name `UniformMemory`
//! (storage buffers), because Unpopped's `Barrier` is "the only ordering a kernel has" and a
//! CUDA `__syncthreads()` orders global writes within the block too; naming only
//! `WorkgroupMemory` would leave a buffer write unordered. `Barrier { Subgroup, Full }` is the same
//! with the `Subgroup` scope. `Fence` is `OpMemoryBarrier`: `Acquire`/`Release`/`AcqRel` map to
//! their SPIR-V semantics, and **`SeqCst` collapses to `AcquireRelease`** (Vulkan's SPIR-V
//! environment has no sequentially consistent barrier). The `Device` fence scope is `Device`.
//!
//! A barrier under thread-divergent control is refused by `check_kernel` (`BarrierDivergent`), which
//! runs first, so it is a typed decline carrying the checker's answer and is never silently moved.

mod atomic;
mod check;
mod declare;
mod flow;

use std::cell::RefCell;
use std::collections::HashMap;

use rspirv::binary::{Assemble, Disassemble};
use rspirv::spirv::{
    ExecutionMode, ExecutionModel, FunctionControl, MemorySemantics, Scope as SpvScope,
    StorageClass, Word,
};
use unpopped::backend::{Decline, DeclinedOp, GeneratedKernel, LowerError, Spelling};
use unpopped::exec::{ExecId, MemorySpace, Scope};
use unpopped::ir::ScalarExpr;
use unpopped::kernel::{BarrierKind, Block, KernelFn, MemoryOrder, Place, Stmt};
use unpopped::types::Ty;

use crate::family::Family;
use crate::ops::{Em, lo};
use crate::{KernelSeams, lower_body};
use check::check_scope;
use declare::{Module, declare};

/// A typed refusal of the kernel layer: `UnsupportedKernel` carrying `what`.
fn refuse(kernel: &KernelFn, what: impl std::fmt::Display) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!("spirv backend: {what} (kernel `{}`)", kernel.name),
    }
}

/// The variant name of a statement: the `Debug` text up to the first non-identifier character.
fn stmt_name(s: &Stmt) -> String {
    let text = format!("{s:?}");
    text.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect()
}

/// Decorate every float add, subtract and multiply `NoContraction`, so a driver cannot fuse a
/// multiply and an add into an FMA the source does not ask for (fuel#568). The Unpopped oracle rounds a
/// statement once at f64, so it cannot tell fused from unfused: the evidence is the instruction count
/// and a host `f32` reference.
fn no_contraction(m: &mut rspirv::dr::Module) {
    use rspirv::dr::{Instruction, Operand};
    use rspirv::spirv::{Decoration, Op};
    let ids: Vec<Word> = m
        .functions
        .iter()
        .flat_map(|f| &f.blocks)
        .flat_map(|b| &b.instructions)
        .filter(|i| matches!(i.class.opcode, Op::FAdd | Op::FSub | Op::FMul))
        .filter_map(|i| i.result_id)
        .collect();
    for id in ids {
        m.annotations.push(Instruction::new(
            Op::Decorate,
            None,
            None,
            vec![
                Operand::IdRef(id),
                Operand::Decoration(Decoration::NoContraction),
            ],
        ));
    }
}

/// A `u32` id value as the compute dtype: a bitcast at 32 bits. At 64 bits it stays the `u32` the IR
/// types an execution id as; the kernel reaches its dtype through a `Cast`.
fn widen(e: &mut Em, u32_t: Word, x: Word) -> Result<Word, LowerError> {
    let t = e.t;
    if t == u32_t || e.fam.keeps_u32_ids() {
        Ok(x)
    } else {
        lo(e.b.bitcast(t, None, x))
    }
}

/// Everything the statement walk needs: the expression emitter and the module-level ids.
struct Kc<'k> {
    kernel: &'k KernelFn,
    em: RefCell<Em>,
    u32_t: Word,
    vec3: Word,
    local_id: Word,
    group_id: Word,
    /// `[SubgroupLocalInvocationId, SubgroupId, SubgroupSize]` when the launch names a subgroup range.
    subgroup: Option<[Word; 3]>,
    /// One storage-buffer variable per parameter.
    bufs: Vec<Word>,
    /// The SPIR-V element type of each parameter's buffer.
    buf_elem: Vec<Word>,
    /// The SPIR-V type of each local (the compute dtype, or an 8/16-bit one).
    local_ty: Vec<Word>,
    /// One workgroup variable per allocation.
    allocs: Vec<Word>,
    /// One function-scope variable per local.
    locals: Vec<Word>,
    block: [u32; 3],
    grid: [u32; 3],
    /// `(merge, continue)` labels of the enclosing loops, innermost last.
    loops: RefCell<Vec<(Word, Word)>>,
    /// Whether `Int64Atomics` has been declared (a 64-bit atomic was lowered).
    wide_atomics: std::cell::Cell<bool>,
}

impl Kc<'_> {
    fn t(&self) -> Word {
        self.em.borrow().t
    }

    /// A `u32` constant.
    fn u32_const(&self, v: u32) -> Word {
        self.em.borrow_mut().b.constant_bit32(self.u32_t, v)
    }

    /// The `axis` component of a builtin vector as the compute dtype.
    fn builtin_axis(&self, var: Word, axis: u8) -> Result<Spelling, LowerError> {
        let mut e = self.em.borrow_mut();
        let v = lo(e.b.load(self.vec3, None, var, None, []))?;
        let x =
            lo(e.b.composite_extract(self.u32_t, None, v, [u32::from(axis)]))?;
        let x = widen(&mut e, self.u32_t, x)?;
        Ok(Spelling::Spelled(x.to_string()))
    }

    /// The subgroup builtin `which` (lane, subgroup index, size) as the compute dtype.
    fn subgroup_builtin(&self, which: usize) -> Result<Spelling, LowerError> {
        let Some(var) = self.subgroup.map(|v| v[which]) else {
            return Ok(Spelling::Declined(Decline::UnsupportedOp {
                op: DeclinedOp::Kernel,
                why: "a subgroup id needs a launch that names a subgroup size range".to_string(),
            }));
        };
        let mut e = self.em.borrow_mut();
        let x = lo(e.b.load(self.u32_t, None, var, None, []))?;
        let x = widen(&mut e, self.u32_t, x)?;
        Ok(Spelling::Spelled(x.to_string()))
    }

    /// The spelling of an execution id.
    fn exec(&self, id: ExecId) -> Result<Spelling, LowerError> {
        let axis = |a: u8| (a < 3).then_some(usize::from(a));
        match id {
            ExecId::ThreadIdx(a) if axis(a).is_some() => self.builtin_axis(self.local_id, a),
            ExecId::BlockIdx(a) if axis(a).is_some() => self.builtin_axis(self.group_id, a),
            ExecId::BlockDim(a) | ExecId::GridDim(a) if axis(a).is_some() => {
                let v = if matches!(id, ExecId::BlockDim(_)) {
                    self.block
                } else {
                    self.grid
                };
                let dim = v[usize::from(a)];
                if self.em.borrow().fam.keeps_u32_ids() {
                    // An execution id is a `u32`; a wide or float kernel keeps it so.
                    return Ok(Spelling::Spelled(self.u32_const(dim).to_string()));
                }
                let dtype = self.em.borrow().dtype;
                Ok(self.em.borrow_mut().constant_int(i64::from(dim), dtype))
            }
            ExecId::LaneId => self.subgroup_builtin(0),
            ExecId::SubgroupId => self.subgroup_builtin(1),
            ExecId::SubgroupSize => self.subgroup_builtin(2),
            other => Ok(Spelling::Declined(Decline::UnsupportedOp {
                op: DeclinedOp::Kernel,
                why: format!("execution id {other:?} is not lowered by the spirv backend v1"),
            })),
        }
    }

    /// The value of local `i`.
    fn local(&self, i: u32) -> Result<Spelling, LowerError> {
        let mut e = self.em.borrow_mut();
        let slot = usize::try_from(i).expect("local id fits usize");
        let (t, var) = (self.local_ty[slot], self.locals[slot]);
        Ok(Spelling::Spelled(
            lo(e.b.load(t, None, var, None, []))?.to_string(),
        ))
    }

    /// Lower an expression to its result id.
    fn expr(&self, e: &unpopped::ir::ScalarExpr) -> Result<Word, LowerError> {
        let tys: Vec<Ty> = self.kernel.locals.iter().map(|l| l.ty()).collect();
        let (local, exec) = (|i| self.local(i), |id| self.exec(id));
        let dtype = self.em.borrow().dtype;
        let seams = KernelSeams {
            local: &local,
            exec: &exec,
            tys: &tys,
            dtype,
        };
        lower_body(&self.em, e, Some(&seams))
    }

    /// The SPIR-V type of the element `place` names.
    fn place_ty(&self, place: &Place) -> Word {
        match place {
            Place::Logical { buffer, .. } => {
                self.buf_elem[usize::try_from(*buffer).expect("buffer id fits usize")]
            }
            _ => self.t(),
        }
    }

    /// A pointer to the element `place` names.
    fn pointer(&self, place: &Place) -> Result<Word, LowerError> {
        let t = self.place_ty(place);
        match place {
            Place::Logical { buffer, coords } if coords.len() == 1 => {
                let idx = self.expr(&coords[0])?;
                let zero = self.u32_const(0);
                let mut e = self.em.borrow_mut();
                let ptr_t = e.b.type_pointer(None, StorageClass::StorageBuffer, t);
                let base = self.bufs[usize::try_from(*buffer).expect("buffer id fits usize")];
                lo(e.b.access_chain(ptr_t, None, base, [zero, idx]))
            }
            Place::Alloc { alloc, index } => {
                let idx = self.expr(index)?;
                let slot = usize::try_from(*alloc).expect("alloc id fits usize");
                let class = match self.kernel.allocs[slot].space {
                    MemorySpace::Private => StorageClass::Private,
                    _ => StorageClass::Workgroup,
                };
                let mut e = self.em.borrow_mut();
                let ptr_t = e.b.type_pointer(None, class, t);
                let base = self.allocs[slot];
                lo(e.b.access_chain(ptr_t, None, base, [idx]))
            }
            other => Err(refuse(
                self.kernel,
                format!("place {other:?} is not lowered (v1: one-coordinate global or shared)"),
            )),
        }
    }

    fn store_local(&self, local: u32, value: Word) -> Result<(), LowerError> {
        let var = self.locals[usize::try_from(local).expect("local id fits usize")];
        lo(self.em.borrow_mut().b.store(var, value, None, []))
    }

    /// `OpControlBarrier` at `scope` ordering workgroup and buffer memory.
    fn barrier(&self, scope: SpvScope) -> Result<(), LowerError> {
        let sem = MemorySemantics::ACQUIRE_RELEASE
            | MemorySemantics::WORKGROUP_MEMORY
            | MemorySemantics::UNIFORM_MEMORY;
        let (s, m) = (self.u32_const(scope as u32), self.u32_const(scope as u32));
        let sem = self.u32_const(sem.bits());
        lo(self.em.borrow_mut().b.control_barrier(s, m, sem))
    }

    /// `OpMemoryBarrier` at `scope` with `order` (`SeqCst` collapses to `AcquireRelease`).
    fn fence(&self, scope: Scope, order: MemoryOrder) -> Result<(), LowerError> {
        let spv = match scope {
            Scope::Subgroup => SpvScope::Subgroup,
            Scope::Block => SpvScope::Workgroup,
            Scope::Device => SpvScope::Device,
            other => {
                return Err(refuse(
                    self.kernel,
                    format!("a fence at scope {other:?} is not lowered"),
                ));
            }
        };
        let ord = match order {
            MemoryOrder::Acquire => MemorySemantics::ACQUIRE,
            MemoryOrder::Release => MemorySemantics::RELEASE,
            MemoryOrder::AcqRel | MemoryOrder::SeqCst => MemorySemantics::ACQUIRE_RELEASE,
            other => {
                return Err(refuse(
                    self.kernel,
                    format!("a {other:?} fence is not lowered"),
                ));
            }
        };
        let sem = ord | MemorySemantics::WORKGROUP_MEMORY | MemorySemantics::UNIFORM_MEMORY;
        let (s, sem) = (self.u32_const(spv as u32), self.u32_const(sem.bits()));
        lo(self.em.borrow_mut().b.memory_barrier(s, sem))
    }

    fn stmt(&self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { .. } | Stmt::Assign { .. } | Stmt::Load { .. } | Stmt::Store { .. } => {
                self.data(s)
            }
            Stmt::MaskedLoad { .. } | Stmt::MaskedStore { .. } => self.masked(s),
            Stmt::Barrier { .. } | Stmt::Fence { .. } => self.sync(s),
            Stmt::AtomicRmw { .. } | Stmt::AtomicCas { .. } => self.atomic(s),
            Stmt::If { .. }
            | Stmt::For { .. }
            | Stmt::While { .. }
            | Stmt::Break
            | Stmt::Continue
            | Stmt::Return => self.flow(s),
            other => Err(refuse(
                self.kernel,
                format!(
                    "`Stmt::{}` is not lowered (v1: Let, Load, Assign, Store, MaskedLoad, MaskedStore, block/subgroup Barrier, Fence, 32-bit AtomicRmw/AtomicCas, control flow)",
                    stmt_name(other)
                ),
            )),
        }
    }

    /// `Let`, `Assign`, `Load` and `Store`.
    fn data(&self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Let { local, init: value } | Stmt::Assign { local, value } => {
                self.assign(*local, value)
            }
            Stmt::Load { local, place } => self.load_stmt(*local, place),
            Stmt::Store { place, value } => self.store_stmt(place, value),
            _ => unreachable!("`stmt` routes only data statements here"),
        }
    }

    /// Predicated access (ROADMAP 9e) as the structured `If` it means: the access sits in the taken arm only, so the
    /// memory is not touched where the mask is false (no bounds check, no race record). A `OpSelect` of a load would
    /// still execute it.
    fn masked(&self, s: &Stmt) -> Result<(), LowerError> {
        let (mask, then, els) = match s {
            Stmt::MaskedLoad {
                local,
                place,
                mask,
                fill,
            } => (
                mask,
                Block(vec![Stmt::Load {
                    local: *local,
                    place: place.clone(),
                }]),
                Block(vec![Stmt::Assign {
                    local: *local,
                    value: fill.clone(),
                }]),
            ),
            Stmt::MaskedStore { place, value, mask } => (
                mask,
                Block(vec![Stmt::Store {
                    place: place.clone(),
                    value: value.clone(),
                }]),
                Block(vec![]),
            ),
            _ => unreachable!("`stmt` routes only masked statements here"),
        };
        self.lower_if(mask, &then, &els)
    }

    fn assign(&self, local: u32, value: &ScalarExpr) -> Result<(), LowerError> {
        let v = self.expr(value)?;
        self.store_local(local, v)
    }

    fn load_stmt(&self, local: u32, place: &Place) -> Result<(), LowerError> {
        let ptr = self.pointer(place)?;
        let t = self.place_ty(place);
        let v = lo(self.em.borrow_mut().b.load(t, None, ptr, None, []))?;
        self.store_local(local, v)
    }

    fn store_stmt(&self, place: &Place, value: &ScalarExpr) -> Result<(), LowerError> {
        let ptr = self.pointer(place)?;
        let v = self.expr(value)?;
        lo(self.em.borrow_mut().b.store(ptr, v, None, []))
    }

    /// `Barrier` (block or subgroup, full) and `Fence`.
    fn sync(&self, s: &Stmt) -> Result<(), LowerError> {
        match s {
            Stmt::Barrier {
                scope: Scope::Block,
                kind: BarrierKind::Full,
            } => self.barrier(SpvScope::Workgroup),
            Stmt::Barrier {
                scope: Scope::Subgroup,
                kind: BarrierKind::Full,
            } => self.barrier(SpvScope::Subgroup),
            Stmt::Fence { scope, order } => self.fence(*scope, *order),
            other => Err(refuse(
                self.kernel,
                format!(
                    "`Stmt::{}` at this scope or kind is not lowered (v1: full Barrier at block or subgroup scope)",
                    stmt_name(other)
                ),
            )),
        }
    }
}

/// A scalar parameter is a specialization constant: its local takes that value at entry.
fn init_spec_locals(m: &mut Module, locals: &[Word]) -> Result<(), LowerError> {
    for &(local, spec) in &m.specs {
        let v = if m.t == m.u32_t {
            spec
        } else if m.fam.is_wide() {
            lo(m.b.u_convert(m.t, None, spec))?
        } else {
            lo(m.b.bitcast(m.t, None, spec))?
        };
        let var = locals[usize::try_from(local).expect("local id fits usize")];
        lo(m.b.store(var, v, None, []))?;
    }
    Ok(())
}

/// Open the entry function and declare one function-scope variable per local, then build the
/// statement emitter over the module.
fn begin_body<'k>(
    kernel: &'k KernelFn,
    mut m: Module,
    block: [u32; 3],
    grid: [u32; 3],
) -> Result<(Kc<'k>, Word), LowerError> {
    let zero_u32 = m.b.constant_bit32(m.u32_t, 0);
    let main =
        lo(m.b.begin_function(m.void, None, FunctionControl::NONE, m.fn_ty))?;
    lo(m.b.begin_block(None))?;
    let local_ty: Vec<Word> = kernel
        .locals
        .iter()
        .map(|l| match Family::of(l.elem) {
            Some(f) if f != m.fam => crate::skeleton::element_type(&mut m.b, f, m.u32_t),
            _ => m.t,
        })
        .collect();
    let locals: Vec<Word> = local_ty
        .iter()
        .map(|&ty| {
            let ptr = m.b.type_pointer(None, StorageClass::Function, ty);
            m.b.variable(ptr, None, StorageClass::Function, None)
        })
        .collect();
    init_spec_locals(&mut m, &locals)?;
    let em = RefCell::new(Em {
        b: m.b,
        fam: m.fam,
        dtype: kernel.compute,
        glsl: m.glsl,
        t: m.t,
        boolean: m.boolean,
        zero_u32,
        inputs: Vec::new(),
        index: zero_u32,
        temps: HashMap::new(),
        strict: true,
        rte: [false; 3],
    });
    let kc = Kc {
        kernel,
        em,
        u32_t: m.u32_t,
        vec3: m.vec3,
        local_id: m.local_id,
        group_id: m.group_id,
        subgroup: m.subgroup,
        bufs: m.bufs,
        buf_elem: m.buf_elem,
        local_ty,
        allocs: m.allocs,
        locals,
        block,
        grid,
        loops: RefCell::new(Vec::new()),
        wide_atomics: std::cell::Cell::new(false),
    };
    Ok((kc, main))
}

/// Lower `kernel` to a SPIR-V module, or say why this slice cannot.
pub(crate) fn lower(kernel: &KernelFn) -> Result<GeneratedKernel, LowerError> {
    let (fam, block, grid) = check_scope(kernel)?;
    let (kc, main) = begin_body(kernel, declare(kernel, fam), block, grid)?;
    let ended = kc.block(&kernel.body)?;
    let mut interface = vec![kc.local_id, kc.group_id];
    interface.extend(kc.subgroup.into_iter().flatten());
    let mut e = kc.em.into_inner();
    if !ended {
        lo(e.b.ret())?;
    }
    lo(e.b.end_function())?;
    e.b.entry_point(ExecutionModel::GLCompute, main, &kernel.name, interface);
    e.b.execution_mode(main, ExecutionMode::LocalSize, block);
    let fam = e.fam;
    if fam.is_float() {
        // The oracle is IEEE: denormals honoured, NaN/Inf/signed zero propagated. Declared at every
        // float width the kernel uses (its compute dtype, and an `F64` buffer or local in an `F32`
        // kernel); a device that cannot honour them must refuse the pipeline (see
        // `required_features`).
        for width in check::float_widths(kernel, fam) {
            e.b.execution_mode(main, ExecutionMode::DenormPreserve, [width]);
            e.b.execution_mode(main, ExecutionMode::SignedZeroInfNanPreserve, [width]);
        }
        e.declare_rte(main);
    }
    let mut module = e.b.module();
    if fam.is_float() {
        no_contraction(&mut module);
    }
    let source = module.disassemble();
    let words = module.assemble();
    Ok(GeneratedKernel::new(kernel.name.clone(), source).with_words(words))
}
