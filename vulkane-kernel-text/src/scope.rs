// SPDX-License-Identifier: MIT OR Apache-2.0
//! The checks that decide whether the walker can lower a kernel, before any text is written.

use std::collections::BTreeSet;

use unpopped::backend::LowerError;
use unpopped::exec::{Extent, MemorySpace};
use unpopped::kernel::{BufferRole, KernelFn, ParamKind, Place, Stmt, check_kernel};
use unpopped_vocab::ElementKind;

use crate::dialect::{Decl, Dialect, Layout};

/// What the checks learned about a kernel.
#[derive(Debug)]
pub(crate) struct Scope {
    pub(crate) signed: bool,
    pub(crate) block: [u32; 3],
    pub(crate) grid: [u32; 3],
    pub(crate) buffers: Vec<Decl>,
    pub(crate) shared: Vec<Decl>,
    pub(crate) subgroup: bool,
    /// The names (`b0`, `a1`) of the buffers and arrays that are atomically typed.
    pub(crate) atomic: BTreeSet<String>,
    pub(crate) backend: &'static str,
}

/// A typed refusal of the kernel layer carrying `what`.
pub(crate) fn refuse(backend: &str, kernel: &KernelFn, what: impl std::fmt::Display) -> LowerError {
    LowerError::UnsupportedKernel {
        detail: format!("{backend} backend: {what} (kernel `{}`)", kernel.name),
    }
}

impl Scope {
    /// The prologue's view of this scope.
    pub(crate) fn layout(&self, kernel: &KernelFn) -> Layout {
        Layout {
            name: kernel.name.clone(),
            signed: self.signed,
            block: self.block,
            buffers: self.buffers.clone(),
            shared: self.shared.clone(),
            spec_locals: self
                .buffers
                .iter()
                .filter_map(|b| b.spec_local.map(|l| (l, b.name.clone())))
                .collect(),
            locals: kernel.locals.len(),
            subgroup: self.subgroup,
        }
    }
}

/// The dtype of the kernel: `I32` or `U32`.
fn dtype(d: &dyn Dialect, kernel: &KernelFn) -> Result<bool, LowerError> {
    match kernel.compute {
        ElementKind::I32 => Ok(true),
        ElementKind::U32 => Ok(false),
        ElementKind::I64 if d.wide() => Ok(true),
        ElementKind::U64 if d.wide() => Ok(false),
        other => Err(refuse(
            d.name(),
            kernel,
            if matches!(other, ElementKind::I64 | ElementKind::U64) {
                format!(
                    "v1 lowers integer kernels only (I32/U32 here: this dialect has no 64-bit integer spelling); compute dtype is {other:?}"
                )
            } else {
                format!("v1 lowers integer kernels only; compute dtype is {other:?}")
            },
        )),
    }
}

/// One buffer parameter, or a decline.
fn buffer(d: &dyn Dialect, kernel: &KernelFn, i: usize) -> Result<Decl, LowerError> {
    let p = &kernel.params[i];
    match &p.kind {
        ParamKind::Buffer { elem, role, rank }
            if *elem == kernel.compute && rank.is_none_or(|r| r == 1) =>
        {
            Ok(Decl {
                name: format!("b{i}"),
                binding: u32::try_from(i).expect("binding fits u32"),
                role: *role,
                len: 0,
                private: false,
                atomic: false,
                spec_local: None,
                len_spec: None,
            })
        }
        ParamKind::Scalar { ty, local }
            if *ty == kernel.compute && sizes_array(kernel, *local) && !d.wide() =>
        {
            Ok(Decl {
                name: format!("s{i}"),
                binding: u32::try_from(i).expect("spec id fits u32"),
                role: BufferRole::Read,
                len: 0,
                private: false,
                atomic: false,
                spec_local: Some(*local),
                len_spec: None,
            })
        }
        _ => Err(refuse(
            d.name(),
            kernel,
            format!(
                "parameter `{}` is not a flat buffer of the compute dtype",
                p.name
            ),
        )),
    }
}

/// Whether some allocation takes its extent from the scalar parameter declaring `local`.
fn sizes_array(kernel: &KernelFn, local: u32) -> bool {
    kernel
        .allocs
        .iter()
        .any(|a| a.extent == Extent::Param(local))
}

/// The identifier of the specialization constant the scalar parameter declaring `local` becomes.
fn spec_name(kernel: &KernelFn, local: u32) -> Option<String> {
    kernel.params.iter().position(|p| {
        matches!(&p.kind, ParamKind::Scalar { ty, local: l } if *l == local && *ty == kernel.compute)
    }).map(|i| format!("s{i}"))
}

/// A shared array whose extent is the scalar parameter declaring `local`, or a decline.
fn dynamic_shared(
    d: &dyn Dialect,
    kernel: &KernelFn,
    i: usize,
    local: u32,
) -> Result<Decl, LowerError> {
    let Some(name) = spec_name(kernel, local) else {
        return Err(refuse(
            d.name(),
            kernel,
            format!(
                "allocation `{}` has a dynamic extent from local {local}, which is no scalar parameter of the compute dtype",
                kernel.allocs[i].name
            ),
        ));
    };
    Ok(Decl {
        name: format!("a{i}"),
        binding: 0,
        role: BufferRole::ReadWrite,
        len: 0,
        private: false,
        atomic: false,
        spec_local: None,
        len_spec: Some(name),
    })
}

/// One shared allocation, or a decline.
fn shared(d: &dyn Dialect, kernel: &KernelFn, i: usize) -> Result<Decl, LowerError> {
    let a = &kernel.allocs[i];
    match (a.space, a.extent) {
        (MemorySpace::Shared | MemorySpace::Private, Extent::Static(n))
            if a.elem == kernel.compute =>
        {
            Ok(Decl {
                name: format!("a{i}"),
                binding: 0,
                role: BufferRole::ReadWrite,
                len: n,
                private: a.space == MemorySpace::Private,
                atomic: false,
                spec_local: None,
                len_spec: None,
            })
        }
        (MemorySpace::Shared, Extent::Param(l)) if a.elem == kernel.compute => {
            dynamic_shared(d, kernel, i, l)
        }
        (MemorySpace::Constant, _) => Err(refuse(
            d.name(),
            kernel,
            format!(
                "allocation `{}` is in the Constant space, which an allocation cannot initialise (ROADMAP 8k: a constant table needs an initialiser the IR does not carry)",
                a.name
            ),
        )),
        _ => Err(refuse(
            d.name(),
            kernel,
            format!(
                "allocation `{}` is not a static shared or private array of the compute dtype (a shared array may have a dynamic extent from a scalar parameter)",
                a.name
            ),
        )),
    }
}

/// Whether `place` is one the walker spells (one coordinate, or a shared element).
pub(crate) fn place_ok(place: &Place) -> bool {
    match place {
        Place::Logical { coords, .. } => coords.len() == 1,
        Place::Alloc { .. } => true,
        _ => false,
    }
}

/// The name the walker gives the buffer or array `place` addresses (`b0`, `a1`).
pub(crate) fn place_name(place: &Place) -> Option<String> {
    match place {
        Place::Logical { buffer, .. } => Some(format!("b{buffer}")),
        Place::Alloc { alloc, .. } => Some(format!("a{alloc}")),
        _ => None,
    }
}

/// The names of the buffers and arrays an atomic statement (at any depth) addresses.
fn atomic_places(block: &unpopped::kernel::Block, into: &mut BTreeSet<String>) {
    for s in &block.0 {
        match s {
            Stmt::AtomicRmw { place, .. } | Stmt::AtomicCas { place, .. } => {
                into.extend(place_name(place));
            }
            Stmt::If { then, els, .. } => {
                atomic_places(then, into);
                atomic_places(els, into);
            }
            Stmt::For { body, .. } | Stmt::While { body, .. } => atomic_places(body, into),
            _ => {}
        }
    }
}

/// Whether any statement (at any depth) is a subgroup barrier or fence.
fn uses_subgroup(block: &unpopped::kernel::Block) -> bool {
    use unpopped::exec::Scope as S;
    block.0.iter().any(|s| match s {
        Stmt::Barrier { scope, .. } | Stmt::Fence { scope, .. } => *scope == S::Subgroup,
        Stmt::If { then, els, .. } => uses_subgroup(then) || uses_subgroup(els),
        Stmt::For { body, .. } | Stmt::While { body, .. } => uses_subgroup(body),
        _ => false,
    })
}

/// The launch shape: present, with a static grid.
fn launch<'k>(
    d: &dyn Dialect,
    kernel: &'k KernelFn,
) -> Result<&'k unpopped::exec::LaunchShape, LowerError> {
    let Some(launch) = kernel.launch.as_ref() else {
        return Err(refuse(
            d.name(),
            kernel,
            "a kernel without a launch shape has no workgroup size",
        ));
    };
    if launch.grid_dims.is_some() {
        return Err(refuse(
            d.name(),
            kernel,
            "a parameter-dependent grid is not lowered",
        ));
    }
    Ok(launch)
}

/// Every local must be a scalar of the compute dtype.
fn locals(d: &dyn Dialect, kernel: &KernelFn) -> Result<(), LowerError> {
    match kernel
        .locals
        .iter()
        .position(|l| l.lanes != 1 || l.elem != kernel.compute)
    {
        Some(i) => Err(refuse(
            d.name(),
            kernel,
            format!("local {i} is not a scalar of the compute dtype"),
        )),
        None => Ok(()),
    }
}

/// A subgroup size range must hold a size a Vulkan device can have: a power of two from 1 to 128. The
/// text targets cannot pin the size either (the host does, from the launch's range), so a range no
/// device could satisfy is refused here by name.
fn subgroup_range(
    d: &dyn Dialect,
    kernel: &KernelFn,
    launch: &unpopped::exec::LaunchShape,
) -> Result<(), LowerError> {
    let Some(sg) = launch.subgroup else {
        return Ok(());
    };
    if sg.min >= 1 && (sg.min..=sg.max.min(128)).any(u32::is_power_of_two) {
        return Ok(());
    }
    Err(refuse(
        d.name(),
        kernel,
        format!(
            "the required subgroup size range {}..={} holds no power of two up to 128, so no Vulkan device can run it",
            sg.min, sg.max
        ),
    ))
}

/// Run every check; see the crate docs for the scope.
pub(crate) fn check(d: &dyn Dialect, kernel: &KernelFn) -> Result<Scope, LowerError> {
    check_kernel(kernel).map_err(|e| LowerError::UnsupportedKernel {
        detail: format!("check_kernel refused kernel `{}`: {e:?}", kernel.name),
    })?;
    let signed = dtype(d, kernel)?;
    let launch = launch(d, kernel)?;
    locals(d, kernel)?;
    subgroup_range(d, kernel, launch)?;
    let mut buffers = (0..kernel.params.len())
        .map(|i| buffer(d, kernel, i))
        .collect::<Result<Vec<_>, _>>()?;
    let mut shared = (0..kernel.allocs.len())
        .map(|i| shared(d, kernel, i))
        .collect::<Result<Vec<_>, _>>()?;
    let mut atomic = BTreeSet::new();
    if d.typed_atomics() {
        atomic_places(&kernel.body, &mut atomic);
    }
    for decl in buffers.iter_mut().chain(shared.iter_mut()) {
        decl.atomic = atomic.contains(&decl.name);
    }
    Ok(Scope {
        signed,
        block: launch.block,
        grid: launch.grid,
        buffers,
        shared,
        subgroup: uses_subgroup(&kernel.body) || launch.subgroup.is_some(),
        atomic,
        backend: d.name(),
    })
}
