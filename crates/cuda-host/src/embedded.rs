/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Host-side loading for embedded device artifact bundles.

use crate::ltoir;
pub use cuda_core::embedded::{
    ArtifactCompileOptions, ArtifactPayloadKind, EmbeddedModule, OwnedArtifactBundle,
    artifact_bundles_from_binary_path, artifact_bundles_from_current_exe,
    embedded_modules_from_current_exe,
};
use cuda_core::{CudaContext, CudaModule, DriverError};
use std::sync::Arc;
use thiserror::Error;

#[cfg(any(target_os = "linux", target_os = "android"))]
mod mapped_image;

/// Errors while discovering, building, or loading an embedded CUDA module.
#[derive(Debug, Error)]
pub enum EmbeddedModuleError {
    /// The artifact anchor could not be associated with its original mapped file.
    #[error("cannot read the binary containing the CUDA artifact anchor: {0}")]
    MappedImage(#[source] std::io::Error),

    /// Reading the embedded artifact section failed.
    #[error(transparent)]
    Core(#[from] cuda_core::EmbeddedModuleError),

    /// The named bundle was not present in the binary that was read.
    #[error("embedded CUDA module '{name}' was not found")]
    ModuleNotFound { name: String },

    /// No embedded bundles with loadable payloads were found.
    #[error("no embedded CUDA modules were found")]
    NoModules,

    /// A bundle existed, but it contained no supported payload.
    #[error("embedded CUDA module '{name}' has no supported payload")]
    UnsupportedPayload { name: String },

    /// PTX source could not be parsed or edited safely while merging bundles.
    #[error("embedded CUDA module '{name}' contains invalid PTX: {reason}")]
    InvalidPtx { name: String, reason: String },

    /// NVVM IR or LTOIR payload compilation failed.
    #[error("failed to build embedded CUDA module: {0}")]
    Ltoir(#[from] ltoir::LtoirError),

    /// The CUDA driver rejected the selected module image.
    #[error("failed to load embedded CUDA module: {0}")]
    Driver(#[from] DriverError),
}

/// Load a named embedded artifact bundle from the current executable.
///
/// Cubin and PTX payloads are loaded directly with the CUDA driver. NVVM IR and
/// LTOIR payloads are linked to an in-memory cubin for their original target.
/// A payload built for a standard pre-Blackwell target, such as `sm_86`, may
/// instead be converted to PTX and JIT-compiled by the driver on Blackwell.
pub fn load_embedded_module(
    ctx: &Arc<CudaContext>,
    name: &str,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_from_current_exe()?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

/// Load a named artifact bundle from the binary containing `anchor`.
///
/// Generated non-generic module loaders borrow their artifact anchor, which
/// the linker places in the same image as the bundle. This works for both an
/// executable and a shared library, including a library opened by a relative
/// path before the working directory changes.
///
/// On Linux and Android, this requires readable `/proc/self/maps` and
/// `/proc/self/map_files`. The original mapped file must still be accessible.
/// A missing, replaced, deleted or unreadable image is an error; it never
/// causes a search in another binary. Other operating systems are unsupported.
/// Payload selection and compilation are identical to [`load_embedded_module`].
pub fn load_embedded_module_from_anchor(
    ctx: &Arc<CudaContext>,
    name: &str,
    anchor: &u8,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundle = artifact_bundles_containing(anchor)?
        .into_iter()
        .find(|bundle| bundle.name == name)
        .ok_or_else(|| EmbeddedModuleError::ModuleNotFound {
            name: name.to_string(),
        })?;
    load_bundle(ctx, &bundle)
}

fn artifact_bundles_containing(
    anchor: &u8,
) -> Result<Vec<OwnedArtifactBundle>, EmbeddedModuleError> {
    #[cfg(any(target_os = "linux", target_os = "android"))]
    {
        let bytes = mapped_image::read(anchor).map_err(EmbeddedModuleError::MappedImage)?;
        oxide_artifacts::read_artifact_bundles_from_object_bytes(&bytes)
            .map_err(cuda_core::EmbeddedModuleError::Artifacts)
            .map_err(Into::into)
    }
    #[cfg(not(any(target_os = "linux", target_os = "android")))]
    {
        let _ = anchor;
        Err(EmbeddedModuleError::MappedImage(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "artifact anchor discovery requires Linux or Android procfs",
        )))
    }
}

/// Merge all PTX bundles from the current executable into a single CUDA module.
///
/// When a generic kernel is monomorphized in a consuming crate, its PTX ends
/// up in that crate's bundle rather than the defining crate's bundle. This
/// function gathers every PTX bundle in the binary, strips duplicate header
/// directives (`.version`, `.target`, `.address_size`) from all but the first
/// bundle, concatenates the bodies, and loads the result as one CUDA module.
/// All kernel symbols are therefore available regardless of which crate bundle
/// they were compiled into.
///
/// Bundles with non-PTX payloads (NVVM IR, LTOIR, cubin) are skipped; use
/// `load_embedded_module` for those.
pub fn load_all_ptx_bundles_merged(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    let bundles = artifact_bundles_from_current_exe()?;
    let merged = merge_ptx_bundles(&bundles)?;

    let module = ctx.load_module_from_image(merged.as_bytes())?;
    // Retain the merged module's `.entry` names (a few dozen bytes per
    // kernel). If a later `_TID_` generic-kernel lookup misses while a
    // same-base entry exists under a different hash, the launch paths can
    // then report a host/device type-identity naming divergence instead of an
    // opaque "named symbol not found". See `crate::entry_registry`.
    crate::entry_registry::register_merged_module_entries(&module, &merged);
    Ok(module)
}

/// Merge the PTX payloads of `bundles`, in iteration order, into one PTX
/// module string: the first PTX bundle keeps its `.version` / `.target` /
/// `.address_size` header directives, every later one contributes its body
/// with those directives stripped. Bundles without a PTX payload are skipped.
///
/// This is the pure half of [`load_all_ptx_bundles_merged`], exposed so tests
/// and examples can check order-dependent merge properties: module-scope
/// symbol uniqueness and extern alignment must hold for every bundle order,
/// not just the one the current executable happens to embed (#1277).
pub fn merge_ptx_bundles<'a>(
    bundles: impl IntoIterator<Item = &'a OwnedArtifactBundle>,
) -> Result<String, EmbeddedModuleError> {
    let mut merged = String::new();
    let mut found_any = false;

    // Debug `.file` indices are per-module sequence numbers starting at 1,
    // so concatenated lineinfo/debug bundles redeclare each other's indices
    // and ptxas rejects the module ("Duplicate file index"). Each appended
    // bundle's indices are shifted past the running maximum (#1292).
    let mut file_index_offset = 0;

    for bundle in bundles {
        if let Some(ptx_bytes) = bundle.payload(ArtifactPayloadKind::Ptx) {
            let ptx_str = std::str::from_utf8(ptx_bytes)
                .map_err(|_| EmbeddedModuleError::UnsupportedPayload {
                    name: bundle.name.clone(),
                })?
                .trim_end_matches('\0');

            let strip_headers = found_any;
            let (body, max_file_index) =
                prepare_bundle_body(ptx_str, strip_headers, file_index_offset).map_err(
                    |reason| EmbeddedModuleError::InvalidPtx {
                        name: bundle.name.clone(),
                        reason,
                    },
                )?;
            file_index_offset = file_index_offset.max(max_file_index);
            merged.push_str(&body);
            if !body.ends_with('\n') {
                merged.push('\n');
            }
            found_any = true;
        }
    }

    if !found_any {
        return Err(EmbeddedModuleError::NoModules);
    }
    Ok(merged)
}

/// Rewrite one bundle's PTX for concatenation: strip the per-file header
/// directives (`.version`/`.target`/`.address_size`) unless this is the first
/// bundle, and shift every debug file index (`.file` declarations, `.loc`
/// references, and `.loc ... inlined_at` references) by `file_index_offset`.
///
/// Returns the rewritten text and the highest file index it declares or
/// references after shifting, so the caller can offset the next bundle past
/// it. Bundles without debug info report `0` and are unaffected.
fn prepare_bundle_body(
    ptx: &str,
    strip_headers: bool,
    file_index_offset: u64,
) -> Result<(String, u64), String> {
    let document = ptx_parse::Document::parse(ptx).map_err(|error| error.to_string())?;
    let mut edits = ptx_parse::EditScript::new();
    let mut max_file_index = 0u64;

    for directive in document.directives() {
        match directive.name() {
            ".version" | ".target" | ".address_size" if strip_headers => {
                edits
                    .delete(directive.line_span())
                    .map_err(|error| error.to_string())?;
            }
            // `.file <index> "path"[, timestamp, size]`
            ".file" => {
                let index = shift_file_index(
                    &mut edits,
                    directive.arguments(),
                    directive.arguments_span().start,
                    0,
                    file_index_offset,
                )?;
                max_file_index = max_file_index.max(index);
            }
            // `.loc <index> <line> <column>[, function_name f, inlined_at
            // <index> <line> <column>]` — both indices reference the file
            // table and both must move with it.
            ".loc" => {
                let arguments = directive.arguments();
                let base = directive.arguments_span().start;
                let index = shift_file_index(&mut edits, arguments, base, 0, file_index_offset)?;
                max_file_index = max_file_index.max(index);
                if let Some(position) = arguments.find("inlined_at") {
                    let after = position + "inlined_at".len();
                    let index =
                        shift_file_index(&mut edits, arguments, base, after, file_index_offset)?;
                    max_file_index = max_file_index.max(index);
                }
            }
            _ => {}
        }
    }

    let body = edits.apply(ptx).map_err(|error| error.to_string())?;
    Ok((body, max_file_index))
}

/// Replace the first integer at or after `from` in `arguments` with itself
/// plus `offset`, recording the edit against the whole document (arguments
/// start at document offset `base`). Returns the shifted value.
fn shift_file_index(
    edits: &mut ptx_parse::EditScript,
    arguments: &str,
    base: usize,
    from: usize,
    offset: u64,
) -> Result<u64, String> {
    let bytes = arguments.as_bytes();
    let mut start = from;
    while start < bytes.len() && !bytes[start].is_ascii_digit() {
        start += 1;
    }
    let mut end = start;
    while end < bytes.len() && bytes[end].is_ascii_digit() {
        end += 1;
    }
    if start == end {
        return Err(format!(
            "malformed debug directive: no file index in {arguments:?}"
        ));
    }
    let index: u64 = arguments[start..end]
        .parse()
        .map_err(|_| format!("debug file index out of range in {arguments:?}"))?;
    let shifted = index + offset;
    if offset != 0 {
        edits
            .replace(base + start..base + end, shifted.to_string())
            .map_err(|error| error.to_string())?;
    }
    Ok(shifted)
}

/// Load the first embedded artifact bundle with a supported payload.
pub fn load_first_embedded_module(
    ctx: &Arc<CudaContext>,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    for bundle in artifact_bundles_from_current_exe()? {
        match load_bundle(ctx, &bundle) {
            Ok(module) => return Ok(module),
            Err(EmbeddedModuleError::UnsupportedPayload { .. }) => continue,
            Err(error) => return Err(error),
        }
    }
    Err(EmbeddedModuleError::NoModules)
}

fn load_bundle(
    ctx: &Arc<CudaContext>,
    bundle: &OwnedArtifactBundle,
) -> Result<Arc<CudaModule>, EmbeddedModuleError> {
    if let Some(cubin) = bundle.payload(ArtifactPayloadKind::Cubin) {
        return Ok(ctx.load_module_from_image(cubin)?);
    }

    if let Some(ptx) = bundle.payload(ArtifactPayloadKind::Ptx) {
        return Ok(ctx.load_module_from_image(ptx)?);
    }

    if let Some(nvvm_ir) = bundle.payload(ArtifactPayloadKind::NvvmIr) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::build_cubin_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::build_ptx_from_nvvm_ir_with_compile_options(
                nvvm_ir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    if let Some(ltoir) = bundle.payload(ArtifactPayloadKind::Ltoir) {
        let emitted = target_arch_for_bundle(bundle)?;
        let execution = ltoir::execution_arch_for_context(ctx)?;
        let image = match ltoir::execution_route(&emitted, &execution)? {
            ltoir::ExecutionRoute::Cubin => ltoir::link_ltoir_to_cubin_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
            ltoir::ExecutionRoute::PtxBridge => ltoir::link_ltoir_to_ptx_with_compile_options(
                ltoir,
                &bundle.name,
                &emitted.sm(),
                bundle.compile_options,
            )?,
        };
        return Ok(ctx.load_module_from_image(&image)?);
    }

    Err(EmbeddedModuleError::UnsupportedPayload {
        name: bundle.name.clone(),
    })
}

fn target_arch_for_bundle(
    bundle: &OwnedArtifactBundle,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    let explicit = std::env::var("CUDA_OXIDE_TARGET").ok();
    target_arch_for_bundle_with_explicit(bundle, explicit.as_deref())
}

fn target_arch_for_bundle_with_explicit(
    bundle: &OwnedArtifactBundle,
    explicit_target: Option<&str>,
) -> Result<cuda_artifact_finalizer::CudaArch, ltoir::LtoirError> {
    ltoir::resolve_source_target(concrete_bundle_target(&bundle.target)?, explicit_target)
}

fn concrete_bundle_target(
    target: &str,
) -> Result<Option<cuda_artifact_finalizer::CudaArch>, ltoir::LtoirError> {
    match target {
        // Compatibility for artifacts emitted before concrete NVVM targets
        // were recorded. New bundles must never use these sentinels.
        "libdevice" | "nvvm-ir" => Ok(None),
        target => Ok(Some(target.parse()?)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bundle_with_target(target: &str) -> OwnedArtifactBundle {
        OwnedArtifactBundle {
            name: "demo".to_string(),
            target: target.to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: Vec::new(),
            entries: Vec::new(),
        }
    }

    #[test]
    fn target_arch_uses_bundle_sm_target() {
        assert_eq!(
            concrete_bundle_target("sm_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    fn ptx_bundle(name: &str, ptx: &str) -> OwnedArtifactBundle {
        use oxide_artifacts::OwnedArtifactPayload;
        OwnedArtifactBundle {
            name: name.to_string(),
            target: "sm_80".to_string(),
            compile_options: ArtifactCompileOptions::new(),
            payloads: vec![OwnedArtifactPayload {
                kind: ArtifactPayloadKind::Ptx,
                name: name.to_string(),
                bytes: ptx.as_bytes().to_vec(),
            }],
            entries: Vec::new(),
        }
    }

    /// The merge is order-explicit: exactly one header set (the first PTX
    /// bundle's), every body present in iteration order, non-PTX bundles
    /// skipped. #1277's collision-free-namespace example checks the same
    /// function under both bundle orders at runtime.
    #[test]
    fn merges_bundles_in_iteration_order_with_one_header_set() {
        let first = ptx_bundle(
            "first",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry a() { ret; }\n",
        );
        let second = ptx_bundle(
            "second",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry b() { ret; }\n",
        );
        let skipped = bundle_with_target("sm_80");

        let forward = merge_ptx_bundles([&first, &skipped, &second]).unwrap();
        assert_eq!(forward.matches(".version").count(), 1);
        assert_eq!(forward.matches(".target sm_80").count(), 1);
        assert!(forward.find(".entry a()").unwrap() < forward.find(".entry b()").unwrap());

        let reversed = merge_ptx_bundles([&second, &first]).unwrap();
        assert_eq!(reversed.matches(".version").count(), 1);
        assert!(reversed.find(".entry b()").unwrap() < reversed.find(".entry a()").unwrap());

        assert!(matches!(
            merge_ptx_bundles([&skipped]),
            Err(EmbeddedModuleError::NoModules)
        ));
    }

    #[test]
    fn strips_only_structural_ptx_module_headers() {
        let ptx = "\
// .target sm_1
.version 8.9
.target sm_120a, debug
.address_size 64
.visible .entry kernel() { ret; }
";
        let (body, max_file_index) = prepare_bundle_body(ptx, true, 0).unwrap();
        assert_eq!(body, "// .target sm_1\n.visible .entry kernel() { ret; }\n");
        assert_eq!(max_file_index, 0);
    }

    /// Debug file indices are per-module, so each appended bundle's `.file`
    /// declarations and `.loc` references (including `inlined_at`) must shift
    /// past the running maximum, or ptxas rejects the merged module with
    /// "Duplicate file index" (#1292). The first bundle keeps its indices.
    #[test]
    fn merged_bundles_renumber_debug_file_indices_per_bundle() {
        let first = ptx_bundle(
            "first",
            "\
.version 8.9
.target sm_80
.address_size 64
.file 1 \"a/lib.rs\"
.file 2 \"shared/thread.rs\"
.visible .entry a()
{
\t.loc\t1 10 5
\tret;
}
",
        );
        let second = ptx_bundle(
            "second",
            "\
.version 8.9
.target sm_80
.address_size 64
.file 1 \"b/main.rs\"
.visible .entry b()
{
\t.loc\t1 3 1, function_name $f, inlined_at 1 7 0
\tret;
}
",
        );
        let plain = ptx_bundle(
            "plain",
            ".version 8.9\n.target sm_80\n.address_size 64\n.visible .entry c() { ret; }\n",
        );

        let merged = merge_ptx_bundles([&first, &plain, &second]).unwrap();

        // First bundle's table and references are untouched.
        assert!(merged.contains(".file 1 \"a/lib.rs\""));
        assert!(merged.contains(".file 2 \"shared/thread.rs\""));
        assert!(merged.contains(".loc\t1 10 5"));

        // The debug-free middle bundle does not advance the offset; the
        // second debug bundle continues after the first's maximum of 2 —
        // declaration, `.loc` reference, and `inlined_at` reference alike.
        assert!(merged.contains(".file 3 \"b/main.rs\""));
        assert!(merged.contains(".loc\t3 3 1, function_name $f, inlined_at 3 7 0"));

        // No index is declared twice across the merged module.
        let declared: Vec<&str> = merged
            .lines()
            .filter(|line| line.trim_start().starts_with(".file"))
            .map(|line| line.split_whitespace().nth(1).unwrap())
            .collect();
        let mut deduped = declared.clone();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(
            declared.len(),
            deduped.len(),
            "duplicate .file index: {declared:?}"
        );
    }

    #[test]
    fn target_arch_uses_bundle_compute_target() {
        assert_eq!(
            concrete_bundle_target("compute_90")
                .unwrap()
                .map(|target| target.sm()),
            Some("sm_90".to_string())
        );
    }

    #[test]
    fn target_arch_falls_back_for_non_arch_target() {
        // Older bundles used these names instead of recording a concrete
        // architecture. New bundles always record a validated `sm_*` target.
        for legacy in ["libdevice", "nvvm-ir"] {
            assert_eq!(concrete_bundle_target(legacy).unwrap(), None);
        }
    }

    #[test]
    fn legacy_bundle_without_recorded_target_requires_explicit_target() {
        for sentinel in ["libdevice", "nvvm-ir"] {
            let bundle = bundle_with_target(sentinel);
            let error = target_arch_for_bundle_with_explicit(&bundle, None)
                .expect_err("the original build target is required");
            assert!(matches!(error, ltoir::LtoirError::TargetNotFound));

            let asserted =
                target_arch_for_bundle_with_explicit(&bundle, Some("compute_86")).unwrap();
            assert_eq!(asserted.sm(), "sm_86");
        }
    }

    #[test]
    fn recorded_bundle_target_overrides_explicit_environment_target() {
        let bundle = bundle_with_target("sm_90");
        let selected = target_arch_for_bundle_with_explicit(&bundle, Some("sm_120")).unwrap();
        assert_eq!(selected.sm(), "sm_90");
    }

    #[test]
    fn target_arch_rejects_malformed_bundle_target() {
        assert!(concrete_bundle_target("sm_90x").is_err());
    }

    #[cfg(any(target_os = "linux", target_os = "android"))]
    #[test]
    fn bundles_of_an_executable_anchor_match_the_current_exe() {
        static ANCHOR: u8 = 0;
        let by_anchor = artifact_bundles_containing(&ANCHOR).unwrap();
        let by_current_exe = artifact_bundles_from_current_exe().unwrap();
        assert_eq!(by_anchor, by_current_exe);
    }
}
