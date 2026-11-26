// SPDX-License-Identifier: MIT OR Apache-2.0
//! Module-level declarations for a kernel: types, builtins, storage buffers, workgroup arrays.

use rspirv::dr::Operand;
use rspirv::spirv::{BuiltIn, Capability, Decoration, StorageClass, Word};
use unpopped::exec::{Extent, MemorySpace};
use unpopped::kernel::{BufferRole, KernelFn, ParamKind};
use unpopped_vocab::ElementKind;

use crate::family::Family;
use crate::skeleton::{element_type, header};

/// An `Input` builtin variable of `vec3` type.
pub(super) fn builtin_var(b: &mut rspirv::dr::Builder, vec3: Word, which: BuiltIn) -> Word {
    let ptr = b.type_pointer(None, StorageClass::Input, vec3);
    let v = b.variable(ptr, None, StorageClass::Input, None);
    b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(which)]);
    v
}

/// An `Input` builtin variable of `u32` type.
fn scalar_builtin(b: &mut rspirv::dr::Builder, u32_t: Word, which: BuiltIn) -> Word {
    let ptr = b.type_pointer(None, StorageClass::Input, u32_t);
    let v = b.variable(ptr, None, StorageClass::Input, None);
    b.decorate(v, Decoration::BuiltIn, [Operand::BuiltIn(which)]);
    v
}

/// The subgroup builtins `[SubgroupLocalInvocationId, SubgroupId, SubgroupSize]`, declared (with the
/// `GroupNonUniform` capability) only for a launch that names a subgroup size range, which is what
/// `check_kernel` requires of a kernel reading them.
fn subgroup_builtins(
    b: &mut rspirv::dr::Builder,
    kernel: &KernelFn,
    u32_t: Word,
) -> Option<[Word; 3]> {
    kernel.launch.as_ref()?.subgroup?;
    b.capability(Capability::GroupNonUniform);
    Some(
        [
            BuiltIn::SubgroupLocalInvocationId,
            BuiltIn::SubgroupId,
            BuiltIn::SubgroupSize,
        ]
        .map(|w| scalar_builtin(b, u32_t, w)),
    )
}

/// The storage-buffer variable of a parameter at binding `binding`.
pub(super) fn buffer_var(
    b: &mut rspirv::dr::Builder,
    ptr: Word,
    binding: u32,
    role: BufferRole,
) -> Word {
    let v = b.variable(ptr, None, StorageClass::StorageBuffer, None);
    b.decorate(v, Decoration::DescriptorSet, [Operand::LiteralBit32(0)]);
    b.decorate(v, Decoration::Binding, [Operand::LiteralBit32(binding)]);
    // On the variable, not the block member: rspirv shares one block type across the buffers, and
    // the roles differ per buffer.
    if role == BufferRole::Read {
        b.decorate(v, Decoration::NonWritable, []);
    }
    v
}

/// The 8- and 16-bit capabilities a kernel needs, declared only for the widths it uses: `Int8` /
/// `Int16` for any such buffer element or local, and the storage-buffer access capability (with its
/// The 8- and 16-bit capabilities a kernel needs, declared only for the widths and kinds it uses:
/// `Int8`/`Int16` for such a buffer element or local, `Float16` for a half one, and the storage-buffer
/// access capability (with its extension) for a buffer element. A 32/64-bit kernel declares none.
fn narrow_capabilities(b: &mut rspirv::dr::Builder, kernel: &KernelFn, fam: Family) {
    let buffers = kernel.params.iter().filter_map(|p| match &p.kind {
        ParamKind::Buffer { elem, .. } => Family::of(*elem),
        _ => None,
    });
    let locals = kernel.locals.iter().filter_map(|l| Family::of(l.elem));
    let is = |f: Family, w: u32, half: bool| {
        f.bits() == w && f.is_half() == half && (f.is_narrow() || half)
    };
    let in_buffer = |w: u32, half: bool| buffers.clone().any(|f| is(f, w, half));
    let anywhere =
        |w: u32, half: bool| in_buffer(w, half) || locals.clone().any(|f| is(f, w, half));
    if anywhere(8, false) {
        b.capability(Capability::Int8);
    }
    if anywhere(16, false) {
        b.capability(Capability::Int16);
    }
    if anywhere(16, true) || fam.is_half() {
        b.capability(Capability::Float16);
    }
    if in_buffer(8, false) {
        b.capability(Capability::StorageBuffer8BitAccess);
        b.extension("SPV_KHR_8bit_storage");
    }
    if in_buffer(16, false) || in_buffer(16, true) {
        b.capability(Capability::StorageBuffer16BitAccess);
        b.extension("SPV_KHR_16bit_storage");
    }
}

/// The module-level ids the function body refers to.
pub(super) struct Module {
    pub(super) b: rspirv::dr::Builder,
    pub(super) fam: Family,
    pub(super) glsl: Word,
    pub(super) void: Word,
    pub(super) fn_ty: Word,
    pub(super) boolean: Word,
    pub(super) u32_t: Word,
    pub(super) t: Word,
    pub(super) vec3: Word,
    pub(super) local_id: Word,
    pub(super) group_id: Word,
    /// The subgroup builtins, when the launch names a subgroup size range.
    pub(super) subgroup: Option<[Word; 3]>,
    pub(super) bufs: Vec<Word>,
    /// The SPIR-V element type of each parameter's buffer (`0` for a scalar parameter).
    pub(super) buf_elem: Vec<Word>,
    pub(super) allocs: Vec<Word>,
    /// `(local, spec constant id)` of each scalar parameter lowered as a specialization constant.
    pub(super) specs: Vec<(u32, Word)>,
}

/// What `declare_storage` made.
pub(super) struct Storage {
    bufs: Vec<Word>,
    buf_elem: Vec<Word>,
    allocs: Vec<Word>,
    specs: Vec<(u32, Word)>,
}

/// The storage buffers (one per parameter) and the workgroup arrays (one per allocation).
pub(super) fn declare_storage(
    b: &mut rspirv::dr::Builder,
    kernel: &KernelFn,
    fam: Family,
    u32_t: Word,
    t: Word,
) -> Storage {
    // One run-time array, block and pointer type per element type (rspirv shares identical types,
    // so each is decorated exactly once); a kernel whose buffers are all the compute dtype has one.
    let mut by_elem: Vec<(ElementKind, Word, Word)> = Vec::new();
    let (mut bufs, mut buf_elem, mut specs) = (Vec::new(), Vec::new(), Vec::new());
    for (i, p) in kernel.params.iter().enumerate() {
        let id = u32::try_from(i).expect("parameter index fits u32");
        match &p.kind {
            ParamKind::Buffer { role, elem, .. } => {
                let known = by_elem
                    .iter()
                    .find(|(e, ..)| e == elem)
                    .map(|&(_, p, e)| (p, e));
                let (buf_ptr, el) = known.unwrap_or_else(|| {
                    let f = Family::of(*elem).expect("check_scope admits integer elements");
                    let el = if f == fam {
                        t
                    } else {
                        element_type(b, f, u32_t)
                    };
                    let rt = b.type_runtime_array(el);
                    b.decorate(
                        rt,
                        Decoration::ArrayStride,
                        [Operand::LiteralBit32(f.bytes())],
                    );
                    let blk = b.type_struct([rt]);
                    b.decorate(blk, Decoration::Block, []);
                    b.member_decorate(blk, 0, Decoration::Offset, [Operand::LiteralBit32(0)]);
                    let ptr = b.type_pointer(None, StorageClass::StorageBuffer, blk);
                    by_elem.push((*elem, ptr, el));
                    (ptr, el)
                });
                bufs.push(buffer_var(b, buf_ptr, id, *role));
                buf_elem.push(el);
            }
            ParamKind::Scalar { local, .. } => {
                // A specialization constant: its SpecId is the parameter's index, its default 1.
                let spec = b.spec_constant_bit32(u32_t, 1);
                b.decorate(spec, Decoration::SpecId, [Operand::LiteralBit32(id)]);
                specs.push((*local, spec));
                bufs.push(spec); // placeholder: a scalar parameter has no buffer
                buf_elem.push(0);
            }
            _ => unreachable!("check_scope admits buffers and extent scalars only"),
        }
    }
    let mut allocs = Vec::new();
    for a in &kernel.allocs {
        let len = match a.extent {
            Extent::Static(n) => b.constant_bit32(u32_t, n),
            Extent::Param(l) => specs.iter().find(|(loc, _)| *loc == l).expect("checked").1,
            _ => unreachable!("check_scope admits static and parameter extents only"),
        };
        let class = match a.space {
            MemorySpace::Private => StorageClass::Private,
            _ => StorageClass::Workgroup,
        };
        let arr = b.type_array(t, len);
        let ptr = b.type_pointer(None, class, arr);
        allocs.push(b.variable(ptr, None, class, None));
    }
    Storage {
        bufs,
        buf_elem,
        allocs,
        specs,
    }
}

/// Declare types, builtins, buffers and workgroup arrays for `kernel`.
pub(super) fn declare(kernel: &KernelFn, fam: Family) -> Module {
    let mut b = header(fam);
    narrow_capabilities(&mut b, kernel, fam);
    if fam.is_float() {
        // IEEE NaN, infinity and signed-zero handling, as the oracle (C without -ffast-math) has it.
        b.capability(Capability::SignedZeroInfNanPreserve);
        if fam != Family::F64 && super::check::float_widths(kernel, fam).contains(&64) {
            b.capability(Capability::Float64);
        }
    }
    let glsl = b.ext_inst_import("GLSL.std.450");
    let void = b.type_void();
    let fn_ty = b.type_function(void, []);
    let boolean = b.type_bool();
    let u32_t = b.type_int(32, 0);
    let t = element_type(&mut b, fam, u32_t);
    let vec3 = b.type_vector(u32_t, 3);
    let local_id = builtin_var(&mut b, vec3, BuiltIn::LocalInvocationId);
    let group_id = builtin_var(&mut b, vec3, BuiltIn::WorkgroupId);
    let subgroup = subgroup_builtins(&mut b, kernel, u32_t);
    let Storage {
        bufs,
        buf_elem,
        allocs,
        specs,
    } = declare_storage(&mut b, kernel, fam, u32_t, t);
    Module {
        b,
        fam,
        glsl,
        void,
        fn_ty,
        boolean,
        u32_t,
        t,
        vec3,
        local_id,
        group_id,
        subgroup,
        bufs,
        buf_elem,
        allocs,
        specs,
    }
}
