# A3 · P0 — Merged-state and coverage audit (issue #1235, slice #1236)

**Audit type:** read-only source audit. No file in the checkout was modified, no build was run, no git state-changing command was issued.
**Checkout:** `/Users/0z5a/Documents/infra/cuda-oxide-a100-20260926/oxide-mac`
**HEAD:** `ec4aa4797956534578a1af010f86252a0b6d8626` (`git rev-parse HEAD`); it contains PR #1236's commit `a07f804a75bebbbd617ce3b018fccfc91b486a55` "feat(wmma): add shared-u32 ldmatrix forms (#1236)" (Gabriel Wu, 2026-09-06), verified with `git merge-base --is-ancestor a07f804a HEAD` → exit 0.

**Local execution capability (frozen for this audit):** host is `Darwin … arm64` (`uname -a`). `nvidia-smi`, `nvcc`, `ptxas`, `cargo`, `rustc` are all absent (`which` returns nothing). Therefore **every compile and every device run in this report is `NOT_RUN` by this auditor**, and no cell below may be read as a local measurement.

**Working-tree caveat:** `git status --short` at audit time showed four dirty paths belonging to a *different*, concurrently running task — `crates/mir-lower/src/convert/ops/arithmetic.rs` (M), `crates/mir-lower/tests/lowering_test/main.rs` (M), `crates/mir-lower/tests/lowering_test/shift_representation.rs` (untracked), `crates/rustc-codegen-cuda/examples/redux_shift_regression/` (untracked). **None of the paths cited in this report is among them**; every file read here still matches its committed blob at the frozen SHA.

**Value legend:** `SUPPORTED` = a tree artefact or gate in this checkout establishes it; `UNSUPPORTED` = a tree artefact establishes it fails; `UNVERIFIED` = supported-by-gate but no artefact for this exact cell, or the tree does not settle it; `NOT_RUN` = executable but not executed here.

---

## 1. Merged API inventory (all present; none `MISSING`)

All 15 requested items exist on `main` at the frozen SHA. The twelve `cuda_device::wmma::ldmatrix_*` names (six pointer forms plus six `*_shared_u32` forms) are **hand-written thunks in `wmma.rs`** whose canonical generated definitions are `cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i0025/i0026/i0027/i0028/i0013/i0029` (`crates/cuda-intrinsics/src/generated/abi_v1.rs:4449-4537`), and the six `*_shared_u32` names are **not separate catalog entries** — they are `compatibility` paths on the same six catalog entries as their pointer twins. That is the single most important structural fact for the rest of this audit: address form is *not* a catalog-level distinction, it is resolved entirely inside the lowering boundary.

| # | Name | Public definition `path:line` | Exact Rust signature | Catalog entry (`intrinsics/catalog.json`) |
|---|---|---|---|---|
| 1 | `ldmatrix_x1_shared_u32` | `crates/cuda-device/src/wmma.rs:149` | `pub unsafe fn ldmatrix_x1_shared_u32(shared_addr: u32) -> u32` | `ldmatrix_m8n8_x1_b16`, abi `i0025`, via `compatibility_paths` `intrinsics/catalog.json:81430-81431` |
| 2 | `ldmatrix_x1_trans_shared_u32` | `crates/cuda-device/src/wmma.rs:160` | `pub unsafe fn ldmatrix_x1_trans_shared_u32(shared_addr: u32) -> u32` | `ldmatrix_m8n8_x1_trans_b16`, abi `i0026`, `intrinsics/catalog.json:81697-81698` |
| 3 | `ldmatrix_x2_shared_u32` | `crates/cuda-device/src/wmma.rs:171` | `pub unsafe fn ldmatrix_x2_shared_u32(shared_addr: u32) -> [u32; 2]` | `ldmatrix_m8n8_x2_b16`, abi `i0027`, `intrinsics/catalog.json:81965-81966` |
| 4 | `ldmatrix_x2_trans_shared_u32` | `crates/cuda-device/src/wmma.rs:182` | `pub unsafe fn ldmatrix_x2_trans_shared_u32(shared_addr: u32) -> [u32; 2]` | `ldmatrix_m8n8_x2_trans_b16`, abi `i0028`, `intrinsics/catalog.json:82234-82235` |
| 5 | `ldmatrix_x4_shared_u32` | `crates/cuda-device/src/wmma.rs:193` | `pub unsafe fn ldmatrix_x4_shared_u32(shared_addr: u32) -> [u32; 4]` | `ldmatrix_m8n8_x4_b16`, abi `i0013`, `intrinsics/catalog.json:82504-82505` |
| 6 | `ldmatrix_x4_trans_shared_u32` | `crates/cuda-device/src/wmma.rs:204` | `pub unsafe fn ldmatrix_x4_trans_shared_u32(shared_addr: u32) -> [u32; 4]` | `ldmatrix_m8n8_x4_trans_b16`, abi `i0029`, `intrinsics/catalog.json:82830-82831` |
| 7 | `ldmatrix_x1` | `crates/cuda-device/src/wmma.rs:49` | `pub unsafe fn ldmatrix_x1(smem_ptr: *const u32) -> u32` | `ldmatrix_m8n8_x1_b16`, compat path `intrinsics/catalog.json:81430` |
| 8 | `ldmatrix_x1_trans` | `crates/cuda-device/src/wmma.rs:64` | `pub unsafe fn ldmatrix_x1_trans(smem_ptr: *const u32) -> u32` | `ldmatrix_m8n8_x1_trans_b16`, `intrinsics/catalog.json:81697` |
| 9 | `ldmatrix_x2` | `crates/cuda-device/src/wmma.rs:84` | `pub unsafe fn ldmatrix_x2(smem_ptr: *const u32) -> [u32; 2]` | `ldmatrix_m8n8_x2_b16`, `intrinsics/catalog.json:81965` |
| 10 | `ldmatrix_x2_trans` | `crates/cuda-device/src/wmma.rs:99` | `pub unsafe fn ldmatrix_x2_trans(smem_ptr: *const u32) -> [u32; 2]` | `ldmatrix_m8n8_x2_trans_b16`, `intrinsics/catalog.json:82234` |
| 11 | `ldmatrix_x4` | `crates/cuda-device/src/wmma.rs:118` | `pub unsafe fn ldmatrix_x4(smem_ptr: *const u32) -> [u32; 4]` | `ldmatrix_m8n8_x4_b16`, `intrinsics/catalog.json:82504` |
| 12 | `ldmatrix_x4_trans` | `crates/cuda-device/src/wmma.rs:133` | `pub unsafe fn ldmatrix_x4_trans(smem_ptr: *const u32) -> [u32; 4]` | `ldmatrix_m8n8_x4_trans_b16`, `intrinsics/catalog.json:82830` |
| 13 | `cvta_generic_to_shared_u32` | `crates/cuda-device/src/shared.rs:544` | `pub unsafe fn cvta_generic_to_shared_u32(ptr: *const u8) -> u32` | **not in `catalog.json`**; intercepted by def-path at `crates/mir-importer/src/translator/terminator/mod.rs:3299-3314` (width `32`) |
| 14 | `cvta_generic_to_shared_offset` | `crates/cuda-device/src/shared.rs:524` | `pub unsafe fn cvta_generic_to_shared_offset(ptr: *const u8) -> u64` | **not in `catalog.json`**; intercepted at `crates/mir-importer/src/translator/terminator/mod.rs:3284-3298` (width `64`) |
| 15 | `mma_m16n8k16_f32_bf16` | `crates/cuda-device/src/wmma.rs:254` | `pub unsafe fn mma_m16n8k16_f32_bf16(c: [f32; 4], a: [u32; 4], b: [u32; 2]) -> [f32; 4]` | `mma_m16n8k16_f32_bf16`, abi `i0105`, `intrinsics/catalog.json:105511` |

Supporting facts:

- Items 1–6 all carry `#[inline(never)]` and a host-side `unreachable!("… called outside CUDA kernel context")` body (`crates/cuda-device/src/wmma.rs:148-152, 159-163, 170-174, 181-185, 192-196, 203-207`); they are compiler-recognised markers, not callable host functions. Same for items 13–14 (`crates/cuda-device/src/shared.rs:523-527, 543-547`).
- Items 1–12 are all routed to the same six MIR dispatch arms by **both** names, e.g. `crates/mir-importer/src/translator/terminator/intrinsics/generated/ldmatrix.rs:802-804` maps `cuda_intrinsics::__cuda_oxide_intrinsic_abi_v1::i0013` / `cuda_device::wmma::ldmatrix_x4` / `cuda_device::wmma::ldmatrix_x4_shared_u32` to one `LdmatrixOp` build; the ABI-id table is `crates/mir-importer/src/translator/terminator/intrinsics/generated/mod.rs:3858-3877`.
- Overlay source of the compatibility pairs: `intrinsics/overlay/ldmatrix.toml:18, 60, 102, 144, 186, 228`.
- `mma_m16n8k16_f32_bf16` is **not** referenced by any example except a never-launched coverage kernel (`crates/rustc-codegen-cuda/examples/generated_intrinsics/src/main.rs:392`, inside `compile_register_mma` at `:387`, whose doc says "This coverage kernel is not launched by the example" at `:384`). Nothing in `generated_ldmatrix` uses MMA at all.

---

## 2. Coverage table

12 combinations = {x1, x2, x4} × {normal, trans} × {pointer, shared-u32}.
`oracle` = does `generated_ldmatrix` **execute and host-verify** this combination (from §3).

| # | load shape | transpose | address form | LLVM NVPTX backend | libNVVM backend | SM80 compile | real A100 run | oracle |
|---|---|---|---|---|---|---|---|---|
| 1 | x1 | normal | pointer | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 2 | x1 | normal | shared-u32 | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 3 | x1 | trans | pointer | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 4 | x1 | trans | shared-u32 | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 5 | x2 | normal | pointer | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 6 | x2 | normal | shared-u32 | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 7 | x2 | trans | pointer | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 8 | x2 | trans | shared-u32 | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 9 | x4 | normal | pointer | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | **YES** (executed + verified) |
| 10 | x4 | normal | shared-u32 | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | **YES** (executed + verified) |
| 11 | x4 | trans | pointer | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |
| 12 | x4 | trans | shared-u32 | SUPPORTED | SUPPORTED | UNVERIFIED | NOT_RUN | no (compile-only) |

**Summary: 2 of 12 combinations are executed and host-verified; 12 of 12 are present in source. The `SM80 compile` column is `UNVERIFIED` for all 12 and the `real A100 run` column is `NOT_RUN` for all 12.**

### 2.1 Shared evidence keys

- **[E-CAT]** Six catalog entries, one per shape/layout pair: `intrinsics/catalog.json:81396` (`ldmatrix_m8n8_x1_b16`, `i0025`), `:81663` (`…x1_trans_b16`, `i0026`), `:81931` (`…x2_b16`, `i0027`), `:82200` (`…x2_trans_b16`, `i0028`), `:82470` (`…x4_b16`, `i0013`), `:82796` (`…x4_trans_b16`, `i0029`). Every one has `target.hardware = any_of [minimum_sm 75]`, `target.targets = "all"`, `target.minimum_ptx = "6.5"`, and exactly one selection with `asm` `ldmatrix.sync.aligned.m8n8.x<N>[.trans].shared.b16 …` and `predicates = ["Subtarget->getPTXVersion() >= 65", "Subtarget->getSmVersion() >= 75"]` — asm lines `:81406, :81673, :81941, :82210, :82480, :82806`; the `SM>=75` predicate lines are `:81409, :81676, :81944, :82213, :82483, :82809`.
- **[E-GATE]** Compile-time target gate is the same floor for all six: `minimum_ptx = 65` and `hardware = AnyOf([MinimumSm(75)])`, e.g. `crates/cuda-oxide-codegen/src/generated_intrinsic_targets/ldmatrix.rs:33-58` (x4 normal); the selection predicates are mirrored at `:62-65`. `sm_80` satisfies all of these.
- **[E-NVPTX]** Catalog `backend_lowerings[backend = "llvm_nvptx"].mechanism = "typed_nvvm"`, `status = "validated"`, target `minimum_sm 75` (`intrinsics/catalog.json:82580-82581`, and the analogous block in each of the six entries). Evidence record stages: `declaration_canonicalization` / `backend_codegen` / `ptx_assembly` all at `sm_75`+`ptx65`, `outcome = "succeeded"`, `runtime_validation = "unexecuted"` (`intrinsics/evidence/rust-llvm-23.1.0-16696adc.json:346` for x1, `:764` for x4; ptxas-accepted details at `:397` and `:821`). Lowering path: `crates/mir-lower/src/convert/intrinsics/ldmatrix.rs:169-195` `lower_with_llvm_intrinsic` → `llvm.nvvm.ldmatrix.sync.aligned.m8n8.x<N>[.trans].b16.p3` (symbols generated at `crates/mir-lower/src/convert/generated_intrinsics/ldmatrix.rs:187, 198, 209, 220, 231, 242`).
- **[E-LIBNVVM]** Catalog `backend_lowerings[backend = "lib_nvvm"].mechanism = "inline_ptx"` (`intrinsics/catalog.json:82648-82649`), stages `backend_codegen` and `device_link` at `sm_75`+`ptx65`, `outcome = "succeeded"` (`:82729-82731`, `:82744-82746`); evidence source `intrinsics/evidence/cuda-13.3-libnvvm-13.3.33.json:14` (x1) and `:342` (x4). Lowering path: `crates/mir-lower/src/convert/intrinsics/ldmatrix.rs:197-237` `lower_with_inline_ptx`, chosen unconditionally for `IntrinsicBackend::LibNvvm` at `:89-97`.
- **[E-PTR]** Pointer form resolution: `crates/mir-lower/src/convert/intrinsics/ldmatrix.rs:124-144` — an LLVM pointer in address space 3 is used as-is (`:136`); address space 0 gets one `llvm::AddrSpaceCastOp` to p3 (`:137-144`); any other address space is a hard error (`:145-149`). A `u64` operand is rejected (`:126-131`). Verified by unit test `crates/mir-lower/tests/lowering_test/matrix_memory.rs:635-752` (both address spaces 0 and 3, all six shapes, callee arg type asserted p3 at `:667-671`).
- **[E-U32]** shared-u32 resolution: a 32-bit integer operand short-circuits to `SharedAddress::NativeU32` (`crates/mir-lower/src/convert/intrinsics/ldmatrix.rs:115-122`). NVPTX: exactly one `llvm::IntToPtrOp` to p3 at the intrinsic boundary (`:180-185`). libNVVM: the u32 is passed straight into the inline-PTX operand (`:206-214`), so no pointer round-trip. Asserted by `crates/mir-lower/tests/lowering_test/matrix_memory.rs:831-911` (x4 normal, both backends: `ptr_to_int == 0`, `address_space_cast == 0`, NVPTX `int_to_ptr == 1` + 1 call, libNVVM `int_to_ptr == 0` + 1 inline-asm).
- **[E-SM80]** `UNVERIFIED`, not `UNSUPPORTED`. The gate admits `sm_80` ([E-GATE]) but no `sm_80` artefact exists: the recorded evidence profile compiles/assembles at `sm_75` only ([E-NVPTX] `:397`, [E-LIBNVVM]); the only other recorded libNVVM target sets are `compute_75/80/86/89/90/90a` (all **failed** for the x4 typed route, `intrinsics/catalog.json:82699-82701`) and `compute_100/120` (`:82714-82716`); CI's compile-only lane drives `generated_ldmatrix` through `cargo oxide emit-ltoir --arch=sm_90` (`scripts/smoketest.sh:2026-2033` dispatch, `:175` floor 75, `:395-399` `LTOIR_ARCH="sm_90"` when no GPU is detected, and every CI runner is `ubuntu-latest`: `.github/workflows/examples-compile.yml:20, 104`); no `.ptx`/`.cubin` is checked in anywhere under `crates/rustc-codegen-cuda/examples/` (`find -name '*.ptx' -o -name '*.cubin'` → empty); and this host cannot compile at all (see header).
- **[E-A100]** `NOT_RUN`: no GPU is present on this host (`nvidia-smi` absent, header) and no CI workflow executes examples (all runners are `ubuntu-latest`, grep of `.github/workflows/*.yml` for `runs-on`). Nothing in the tree records an A100/SM80 execution of any of the six forms.

### 2.2 One evidence line per row

| # | combination | evidence line |
|---|---|---|
| 1 | x1 / normal / pointer | [E-CAT] `ldmatrix_m8n8_x1_b16`/`i0025` + [E-NVPTX] + [E-LIBNVVM]; [E-PTR] AS0→`addrspacecast` p3, AS3 direct; executed nowhere (§3). |
| 2 | x1 / normal / shared-u32 | Same catalog id via compat path `intrinsics/catalog.json:81431`; [E-U32] NVPTX one `inttoptr` p3 / libNVVM raw `r` operand; executed nowhere. |
| 3 | x1 / trans / pointer | [E-CAT] `ldmatrix_m8n8_x1_trans_b16`/`i0026`; [E-PTR]; inline PTX head `ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16` (`crates/mir-lower/src/convert/generated_intrinsics/ldmatrix.rs:296`). |
| 4 | x1 / trans / shared-u32 | Same id via compat path `intrinsics/catalog.json:81698`; [E-U32]. |
| 5 | x2 / normal / pointer | [E-CAT] `ldmatrix_m8n8_x2_b16`/`i0027`; [E-PTR]; NVPTX callee `…_x2_b16_p3`. |
| 6 | x2 / normal / shared-u32 | Same id via compat path `intrinsics/catalog.json:81966`; [E-U32]. |
| 7 | x2 / trans / pointer | [E-CAT] `ldmatrix_m8n8_x2_trans_b16`/`i0028`; inline PTX head at `crates/mir-lower/src/convert/generated_intrinsics/ldmatrix.rs:334`. |
| 8 | x2 / trans / shared-u32 | Same id via compat path `intrinsics/catalog.json:82235`; [E-U32]. |
| 9 | x4 / normal / pointer | [E-CAT] `ldmatrix_m8n8_x4_b16`/`i0013`; [E-PTR]; **executed and verified** — `crates/rustc-codegen-cuda/examples/generated_ldmatrix/src/main.rs:156,197` call `ldmatrix_m8n8_x4_b16` with a `*const u32` shared pointer, launched at `:344-355` and `:383-395`, checked at `:361-375` and `:401-415`. |
| 10 | x4 / normal / shared-u32 | Same id via compat path `intrinsics/catalog.json:82505`; [E-U32]; **executed and verified** — `main.rs:112` `cvta_generic_to_shared_u32`, `:113` `ldmatrix_x4_shared_u32`, launched `:305-314`, checked `:321-336`. |
| 11 | x4 / trans / pointer | [E-CAT] `ldmatrix_m8n8_x4_trans_b16`/`i0029`; inline PTX head at `crates/mir-lower/src/convert/generated_intrinsics/ldmatrix.rs:372`; executed nowhere. |
| 12 | x4 / trans / shared-u32 | Same id via compat path `intrinsics/catalog.json:82831`; [E-U32]; executed nowhere. |

**Caveat on the libNVVM column.** `SUPPORTED` is the honest value for all 12 rows because the *selected and validated* libNVVM mechanism is `inline_ptx` ([E-LIBNVVM]) and `crates/mir-lower/src/convert/intrinsics/ldmatrix.rs:89-97` always takes that path. But the catalog also records a **failed** libNVVM `typed_nvvm` experiment for `ldmatrix_m8n8_x4_b16` only: `declaration_canonicalization` at `compute_75/80/86/89/90/90a` succeeded (`intrinsics/catalog.json:82680-82682`) while `backend_codegen` there **failed** with `NVVM_ERROR_COMPILATION(9): unsupported operation` (`:82699-82701`), succeeding only at `compute_100/120` (`:82714-82716`). This is a libNVVM *intrinsic-availability* limit, not a cuda-oxide lowering failure, and it does not affect the shipped inline-PTX route — but it is the reason the x4 entry needed a selected inline-PTX fallback at all (`:82729-82731`). The x1/x2 entries carry no such failed stage.

---

## 3. The existing oracle: `generated_ldmatrix`

File: `crates/rustc-codegen-cuda/examples/generated_ldmatrix/src/main.rs` (424 lines). Directory contains only `Cargo.toml`, `Cargo.lock`, `src/main.rs` — **no `README.md`**.

### 3.1 What it defines

Four kernels in `#[cuda_module] mod kernels` (`:63-277`):

| kernel | line | shape / transpose / address form called | launched? | host-verified? |
|---|---|---|---|---|
| `ldmatrix_x4_oracle` | `:68-124` | x4, normal, **shared-u32** — `cvta_generic_to_shared_u32` at `:111-112`, `ldmatrix_x4_shared_u32` at `:113` | yes, `:305-314` | yes, `:321-336` |
| `ldmatrix_x4_dynamic_index_oracle` | `:128-162` | x4, normal, **pointer** — `cuda_intrinsics::matrix::ldmatrix_m8n8_x4_b16` at `:156` with `shared.add(row_word).cast_const()` | yes, `:344-355` | yes, `:361-375` |
| `ldmatrix_x4_assert_bounded_index_oracle` | `:166-208` | x4, normal, **pointer** — same intrinsic at `:197`, dynamic `register_index` argument | yes, `:383-395` | yes, `:401-415` |
| `legacy_ldmatrix_compile_oracle` | `:211-276` | **all six shapes × both layouts × both address forms** — pointer calls at `:231-236`, `cvta_generic_to_shared_u32` at `:237`, u32 calls at `:238-243`, 28 output words per lane at `:247-274` | **never** | **never** |

The kernel names it calls: `ldmatrix_x1`, `ldmatrix_x1_trans`, `ldmatrix_x2`, `ldmatrix_x2_trans`, `ldmatrix_x4`, `ldmatrix_x4_trans` (`:231-236`); `ldmatrix_x1_shared_u32`, `ldmatrix_x1_trans_shared_u32`, `ldmatrix_x2_shared_u32`, `ldmatrix_x2_trans_shared_u32`, `ldmatrix_x4_shared_u32`, `ldmatrix_x4_trans_shared_u32` (`:238-243`); plus `ldmatrix_m8n8_x4_b16` (`:156`, `:197`). Imports are at `:21-31`. **No MMA call of any kind appears in the file.**

### 3.2 What it actually executes and verifies

`fn main()` is `:279-424`. It launches exactly the three x4/normal kernels (`:305`, `:345`, `:384`) and asserts on their output (`:321-336`, `:361-375`, `:401-415`). `legacy_ldmatrix_compile_oracle` is never referenced outside its own definition (`grep -rn legacy_ldmatrix_compile_oracle` over the tree returns only `:212`). Consequence: **ten of the twelve combinations are compiled-but-never-reached, and none of them has any expected-value check.**

The two verified combinations are the two x4/normal rows:

- **x4 / normal / shared-u32** — `main.rs:305-336`. This is the kernel the file's module doc (`:6-16`) describes.
- **x4 / normal / pointer** — `main.rs:344-375` and `:383-415`, through the *canonical* path `cuda_intrinsics::matrix::ldmatrix_m8n8_x4_b16` rather than the `cuda_device::wmma::ldmatrix_x4` alias; the two are the same catalog entry and the same MIR dispatch arm (§1).

### 3.3 Layout, swizzle, offsets, patterns, launch shape

- **Layout / swizzle:** one layout only. `static mut INPUT: SharedArray<u32, 128, 16>` (`:72`, `:129`, `:170`, `:213`), i.e. four contiguous row-major 8×8 b16 matrices in 512 bytes, row stride 16 bytes, **no padding and no swizzle**. `WORDS_PER_ROW = 4`, `ROWS = 8`, `MATRICES = 4`, `SHARED_WORDS = 128` (`:34-38`). The 16-byte alignment is explicit in the `SharedArray` third type parameter (contract at `crates/cuda-device/src/shared.rs:83`, `:116`).
- **Offsets:** tile base is always **0**. Each lane supplies its own row address `shared.add(lane * WORDS_PER_ROW)` (`:84`, `:138`, `:179`, `:221`), i.e. lane `n` → row `n % 8` of matrix `n / 8` (documented at `:108`). There is **no non-zero tile-base offset case and no near-end-of-allocation case.**
- **Patterns:** one pattern family — `matrix_element()` at `:50-52` packs `bit15 marker | bits11:10 matrix | bits7:5 row | bits2:0 column`, i.e. distinguishable per matrix/row/column, but **no `0xffff`, no all-zero, and no seeded-random pattern**. The `legacy` kernel fills every word with `lane as u32` (`:223-226`), which is not lane-discriminating per column.
- **Launch:** `grid_dim (1,1,1)`, `block_dim (32,1,1)`, `shared_mem_bytes: 0` (`:307-311`, `:347-351`, `:386-390`) — **one warp only, no multi-warp case**, and dynamic shared memory is unused.
- **Synchronisation:** one `thread::sync_threads()` between the shared writes and the load (`:103`, `:154`, `:195`, `:228`).

### 3.4 Target architecture and skip behaviour

`main.rs:281-293` queries `compute_capability` and returns early with `println!("PASS (skipped): ldmatrix.m8n8.x4.b16 requires sm_75+; device is sm_{major}{minor}")` when it is `< 75`. The example therefore **requires `sm_75`+, not `sm_80`**, and has no `sm_80`-specific path. Note the trap this creates for coverage claims: the smoketest accepts a `PASS (skipped)` line as PASS (`scripts/smoketest.sh:556-561`, and the comment there names `generated_ldmatrix` explicitly), so on a pre-Turing device this oracle is green without executing a single kernel.

### 3.5 Uncovered cells — explicit list

| uncovered | why |
|---|---|
| x1 / normal / pointer | only inside the never-launched `legacy_ldmatrix_compile_oracle` (`main.rs:231`) |
| x1 / normal / shared-u32 | `main.rs:238` — never launched |
| x1 / trans / pointer | `main.rs:232` — never launched |
| x1 / trans / shared-u32 | `main.rs:239` — never launched |
| x2 / normal / pointer | `main.rs:233` — never launched |
| x2 / normal / shared-u32 | `main.rs:240` — never launched |
| x2 / trans / pointer | `main.rs:234` — never launched |
| x2 / trans / shared-u32 | `main.rs:241` — never launched |
| x4 / trans / pointer | `main.rs:236` — never launched |
| x4 / trans / shared-u32 | `main.rs:243` — never launched |
| x4 / normal / **pointer-fragment-vs-u32 equivalence** | the two verified kernels never compare their fragments against each other; each is checked only against the same host-side `matrix_word()` model (`:325`, `:365`, `:405`) |
| any row, **non-zero tile offset** | every address is derived from tile base 0 (§3.3) |
| any row, **padded / swizzled layout** | single unpadded layout (§3.3) |
| any row, **`mma_m16n8k16_f32_bf16` consumption** | no MMA in the file at all (§3.1) |
| any row, **>1 warp** | `block_dim (32,1,1)`, grid `(1,1,1)` (§3.3) |

**Hazard if `legacy_ldmatrix_compile_oracle` is simply wired up as-is:** it writes `LANES * LEGACY_REGISTERS = 32 * 28 = 896` words (`main.rs:42`, `:245-275`), but `main()` allocates only `OUTPUT_WORDS = LANES * MATRICES = 128` words (`:39`, `:297-298`). `DisjointSlice::get_unchecked_mut` only carries a `debug_assert!` (`crates/cuda-device/src/disjoint.rs:434-441`), so an optimized build would write out of bounds. A follow-up must allocate `LANES * LEGACY_REGISTERS` words and pass the matching slice.

---

## 4. Translation-unit / address-arithmetic contract

### 4.1 How the u32 shared address is produced

| stage | `path:line` | content |
|---|---|---|
| API declaration | `crates/cuda-device/src/shared.rs:544` | `pub unsafe fn cvta_generic_to_shared_u32(ptr: *const u8) -> u32` |
| doc: what the value is | `crates/cuda-device/src/shared.rs:529-535` | "Convert a generic address into its 32-bit `.shared::cta` state-space address… this form matches the `u32` address operand consumed by CTA-shared instructions such as `ldmatrix`. Convert a shared base once, then keep byte-offset and swizzle arithmetic in `u32` when the complete addressed allocation fits the CTA shared window." |
| doc: precondition | `crates/cuda-device/src/shared.rs:537-540` | "`ptr` must be a generic pointer to memory in the current CTA's shared-memory window. Converting any other pointer produces an unspecified address." |
| why not `pub use` | `crates/cuda-device/src/shared.rs:519-522` | the importer intercepts the exact rendered def-path, so a re-export would silently break interception |
| MIR interception, width 32 | `crates/mir-importer/src/translator/terminator/mod.rs:3299-3314` | `cuda_device::shared::cvta_generic_to_shared_u32` → `emit_cvta_generic_to_shared_offset(…, 32)` |
| op built | `crates/mir-importer/src/translator/terminator/intrinsics/memory.rs:1302-1370` | builds `dialect_nvvm::ops::CvtaGenericToSharedOffsetOp` with `IntegerType::get(ctx, result_width, Unsigned)` (`:1348`) |
| lowering | `crates/mir-lower/src/convert/intrinsics/memory.rs:29-56` | `cast_to_shared_addrspace` (`crates/mir-lower/src/convert/intrinsics/common.rs:79-99`: one `llvm::AddrSpaceCastOp` to p3 unless already in p3) then a single `llvm::PtrToIntOp` into an integer of the op's result width (`:51-52`) — i.e. LLVM IR `ptrtoint (addrspacecast (ptr addrspace(0) → ptr addrspace(3)) to i32)`. The IR carries casts, not a `cvta.to.shared` call. |

The `u64` sibling is the same function with `result_width = 64` (`crates/mir-importer/src/translator/terminator/mod.rs:3284-3298`), which is why its doc at `crates/cuda-device/src/shared.rs:500-505` says hardware SMEM descriptors encode `(start_address >> 4) & 0x3FFF` on the space-local offset.

### 4.2 Which lane supplies which row address (x1 / x2 / x4)

The normative contract is in the generated ABI docs, which are more precise than the hand-written `wmma.rs` prose:

| shape | lanes that supply row addresses | replication precondition | `path:line` |
|---|---|---|---|
| x1 | 8 (lanes 0–7) | "For portable sm_75 behavior, otherwise-unused lanes must also carry valid addresses replicated from the contributing lanes." | `crates/cuda-intrinsics/src/generated/abi_v1.rs:4449-4457` |
| x1.trans | 8 (lanes 0–7) | same | `crates/cuda-intrinsics/src/generated/abi_v1.rs:4465-4473` |
| x2 | 16 (lanes 0–15) | same | `crates/cuda-intrinsics/src/generated/abi_v1.rs:4481-4489` |
| x2.trans | 16 (lanes 0–15) | same | `crates/cuda-intrinsics/src/generated/abi_v1.rs:4497-4505` |
| x4 | 32 (all lanes) | none stated | `crates/cuda-intrinsics/src/generated/abi_v1.rs:4513-4521` |
| x4.trans | 32 (all lanes) | none stated | `crates/cuda-intrinsics/src/generated/abi_v1.rs:4528-4536` |

Every one of those six docstrings additionally states: *"All 32 warp lanes must execute the same instruction. PTX maps <8|16|32> lane-provided row addresses for this X<n> variant; addresses may alias, but each used address must be 16-byte aligned and have 16 readable shared-memory bytes."* and *"This weak memory operation does not replace a required barrier or fence."*

The hand-written mirror in `wmma.rs` is at `:11-18` (`x1: lanes 0..7`, `x2: lanes 0..15`, `x4: lanes 0..31`; "On sm_75, x1 and x2 still require valid addresses in all 32 lanes. A common choice is to copy the lower-lane addresses into the upper lanes."; ".trans forms use column-major rather than row-major layout") and per-item at `:43-47` (`ldmatrix_x1`), `:79-82` (`ldmatrix_x2`), `:114-116` (`ldmatrix_x4`).

The generated intrinsic doc for the u32 conversions carries **no** lane table of its own — each `_shared_u32` item simply says "Same safety contract as [`ldmatrix_xN`]" (`crates/cuda-device/src/wmma.rs:145-147`, `:156-158`, `:167-169`, `:178-180`, `:189-191`, `:200-202`). **There is no mask operand anywhere in this API**; the `.sync.aligned` "mask" is the implicit requirement that all 32 lanes execute the same instruction with no exited lanes.

### 4.3 Element-vs-byte offset semantics: documented only on the u32 side

- Byte semantics **are** stated twice for the u32 path: "keep subsequent **byte** arithmetic in `u32`" (`crates/cuda-device/src/wmma.rs:142-143`) and "keep **byte-offset** and swizzle arithmetic in `u32`" (`crates/cuda-device/src/shared.rs:534-535`).
- Byte semantics are also stated for the other shared-offset API: "The offset is in **bytes**, not elements." (`crates/cuda-device/src/shared.rs:466`, for `DynamicSharedArray::offset`), with the alignment consequence at `:484-488`.
- **Missing:** nowhere in `wmma.rs` or `shared.rs` is there a statement that the pointer forms take `*const u32` and therefore that Rust `.add(i)` counts **elements of `u32` (4 bytes each)**, in contrast to the u32 forms' **byte** offsets. `grep -n "element" crates/cuda-device/src/wmma.rs` returns only MMA-fragment prose (`:216-457`), and the pointer-form docs (`:43-47`, `:79-82`, `:114-116`) never mention units. The one example that uses both forms sidesteps the question by converting *after* the element-scaled pointer arithmetic (`crates/rustc-codegen-cuda/examples/generated_ldmatrix/src/main.rs:230-237`, where `shared.add(row_word)` is a `u32*` add and `cvta_generic_to_shared_u32` is applied to the already-offset pointer). **A test that mixes the two forms must therefore do the `× 4` itself; the tree gives it no written rule to cite.**

### 4.4 Preconditions a test must respect

| precondition | `path:line` |
|---|---|
| Each used row address is 16-byte aligned and has 16 readable shared bytes | `crates/cuda-intrinsics/src/generated/abi_v1.rs:4455`, `:4471`, `:4487`, `:4503`, `:4519`, `:4534` (docstring lines, full ranges in §4.2); `crates/cuda-device/src/wmma.rs:43`, `:79`, `:114` |
| `SharedArray` base alignment is the third const generic, in bytes; default 0 = natural alignment | `crates/cuda-device/src/shared.rs:83`, `:116` |
| All 32 lanes must execute the same instruction; no exited lanes (`.sync.aligned`) | `crates/cuda-device/src/wmma.rs:45`, `:81`, `:115`; `crates/cuda-device/src/wmma.rs:245-247` for the MMA analogue |
| x1/x2 on sm_75 also need valid addresses in the otherwise-unused lanes | `crates/cuda-device/src/wmma.rs:20-21`, `:44`, `:80`; generated docstrings above |
| `.sync` orders execution but **not** memory: a barrier/fence is still required around the shared writes | `crates/cuda-device/src/wmma.rs:24-27`; generated docstrings ("This weak memory operation does not replace a required barrier or fence"); the example's `thread::sync_threads()` at `crates/rustc-codegen-cuda/examples/generated_ldmatrix/src/main.rs:101-103` |
| The `u32` address must denote a live region of the **current CTA's** shared window; the whole addressed allocation must fit the CTA shared window | `crates/cuda-device/src/shared.rs:534-535`, `:537-540` |

---

## 5. Example registration mechanics

### 5.1 How an example becomes visible to `cargo oxide list --json`

`crates/cargo-oxide/src/commands/examples_list.rs:31-54` is the whole mechanism: it requires a cuda-oxide checkout (`:32-37`, `is_workspace` from `crates/cargo-oxide/src/commands/context.rs:39, 66, 116`), enumerates **direct subdirectories** of `crates/rustc-codegen-cuda/examples/` (`crates/cargo-oxide/src/commands/context.rs:58, 108`), and emits one entry per directory that contains a top-level `Cargo.toml` with a `[package]` table:

- directory name is the `name` (`examples_list.rs:83-88`, `:128-133`);
- a directory **without** `Cargo.toml` is skipped with a warning, not an error (`:90-99`);
- a `Cargo.toml` without `[package]` is a hard error (`:155-158`);
- `title` / `description` / `requirements` come from `README.md` when present (`:103-110`, parsed at `:170-225`), else `description` falls back to `package.description` (`:160-165`), else the literal `"No description documented."` (`:120-126`);
- JSON shape is `{"schema_version": 1, "examples": [{name, title, description, requirements}]}` (`:599-620`).

Requirements blocks in a README are recognised only under a requirements-style heading (`:552-563`) and only from list items or two-column tables (`:391-483`, `:531-550`).

Crucially, this directory scan is **the same discovery rule** used by the GPU test suite: `scripts/smoketest.sh:498-510` globs `*/Cargo.toml`, and `:512-517` fails the run if any directory under `examples/` lacks a top-level `Cargo.toml`.

### 5.2 Exact set of edits for a new example

Required:

1. **New directory** `crates/rustc-codegen-cuda/examples/<name>/`, where `<name>` must equal the `[package] name` because the smoketest derives artifact filenames from the directory name with hyphens mapped to underscores (`scripts/smoketest.sh:906-912`, comment at `:907-911`).
2. **`Cargo.toml`** with a `[package]` table (`crates/cargo-oxide/src/commands/examples_list.rs:155-158`) and its own `[workspace]` — every example is a standalone workspace, e.g. `crates/rustc-codegen-cuda/examples/generated_ldmatrix/Cargo.toml:9`. It must carry the shared host-crate pins (`cuda-bindings`/`cuda-core`/`cuda-async`) in the **same form and version** as the root manifest, because `scripts/check-shared-crate-pin.sh:87` iterates every `examples/*/Cargo.toml` via `git ls-files`. Copy `crates/rustc-codegen-cuda/examples/generated_ldmatrix/Cargo.toml:1-15` as the template.
3. **`src/main.rs`** beginning with the standard SPDX header inside the first 15 lines (`scripts/check-spdx-headers.sh:52-54`, `HEADER_WINDOW=15` at `:54`, window read at `:67`).
4. **`Cargo.lock`** next to it. `scripts/check-example-license-policy.sh:95-116` asserts "Every example directory that carries a Cargo.toml carries a Cargo.lock beside it" (`:98-108`) and hard-fails otherwise (`:111-116`); `scripts/sync-example-locks.sh:46-48` says to copy an existing example's lock (e.g. `vecadd`'s) before the first build and then run `--check` / `--align`.
5. **A success marker** (`SUCCESS`, `PASS` or `Complete` on a non-comment line, not matching the skip pattern) in `src/**/*.rs` **unless** the example is claimed by one of the smoketest category arrays — `scripts/check-example-smoketest-contract.sh:237-238` (the two regexes), `:306-317` (`CATEGORY_LISTS`), `:348-372` (the unmarked check).
6. **`dependency-licenses.csv` rows only if new third-party crates are introduced** (`scripts/check-dependency-licenses.sh:245-256`; the example half starts at `:258-259`). An example depending only on `cuda-intrinsics`, `cuda-device`, `cuda-host` (path) and `cuda-core` (crates.io, already inventoried) needs no new rows.
7. If a `README.md` is added, any host-API spelling in it is checked by `scripts/check-host-api-paths.sh:42-43` (example READMEs are in the scanned file set) — so use the current spellings from `cuda-core`/`cuda-async`.

Optional / conditional:

8. **Smoketest category membership** only if the new example needs a non-default route. Default is `standard` (`scripts/smoketest.sh:122-135`, `echo standard` at `:134`). To force the libNVVM `emit-ltoir` compile-only route, add it to `NVVM_VERIFY_EXAMPLES` (`scripts/smoketest.sh:85`) and, if it needs an architecture floor, add a `case` arm to `nvvm_verify_arch` (`:165-186`, e.g. `:175`). If added to any category array, `scripts/check-example-smoketest-contract.sh:336-346` enforces that no example is claimed by two arrays.
9. **`verify-debug-info.sh` / `verify-code-shape.sh`** only if you want per-example structural checks; they are looked up by convention at `scripts/smoketest.sh:1206-1207` and `:1278`.

**Not needed (verified, so a follow-up does not waste edits here):**

- Root `Cargo.toml` `members` — examples are not workspace members; the list is explicit and contains no `examples/` entry (`Cargo.toml:3-37`).
- `.github/workflows/examples-compile.yml` — discovery is by glob inside `scripts/smoketest.sh:498-517`; no per-example workflow entry exists (`examples-compile.yml:104` runs the whole suite).
- `scripts/check-test-matrix-coverage.sh` — it compares the root workspace `members` list against `unit-tests.yml`'s `- package:` entries and the Justfile's `-p` flags (`scripts/check-test-matrix-coverage.sh:81-99`, `:102-107`, `:129-139`, `:163-193`). A new **example** is not a workspace member, so this guard needs no change; it is the guard for new workspace **crates**.
- `crates/rustc-codegen-cuda/README.md` — `:134` describes the `examples/` directory in prose without enumerating it.
- `cuda-oxide-book/` — no page enumerates examples; `generated_ldmatrix` appears nowhere in the book.

Validation commands for a new example (all from the task's own §8):

```
bash scripts/check-example-smoketest-contract.sh
bash scripts/check-test-matrix-coverage.sh
cargo oxide fmt --check
cargo oxide build <name> --arch sm_80
```

---

## 6. Gap statement

**Missing cells, cheapest first.** "Cheapest" is ranked by (a) whether the combination's lowering is already proven at `sm_75` on both backends, (b) whether the hardware contract is already documented, and (c) whether the device oracle must run on a real SM80 GPU.

| rank | missing cell | fill cost | needs a real SM80 GPU? |
|---|---|---|---|
| 1 | Rows 1, 3, 5, 7 — the four x1/x2 **pointer** forms — and row 11, the x4 transposed **pointer** form | Lowest. `legacy_ldmatrix_compile_oracle` already contains the calls (`main.rs:231-236`); the only work is to launch it against a correctly-sized buffer (`LANES * LEGACY_REGISTERS = 896` words, §3.5) and add a host-side lane/row/column model. Lowering already proven at `sm_75` on both backends ([E-NVPTX], [E-LIBNVVM]). | Execute: yes. But the expected fragments can be computed and asserted on the **host** in a CPU-only unit test of the same address model, so the *logic* can be pre-verified without a GPU. |
| 2 | Rows 4, 8, 12 — the three **transposed shared-u32** forms — plus rows 2, 6, the x1/x2 **normal shared-u32** forms | Low. Same kernel, same address, `main.rs:238-243` already calls them. The only extra reasoning is the `.trans` fragment mapping, which `wmma.rs:22` and the catalog's `operation_key` (`…x1.transposed.b16.shared`) already state, and which is verified against the same host model. | Execute: yes. |
| 3 | Any row with a **non-zero tile-base offset** or a **near-end-of-allocation tile** | Low–moderate: a new kernel, but no new API and no new lowering. Directly required by the task's P1 matrix (task §4 table, "offset" row). | Execute: yes. |
| 4 | Any row with a **padded or swizzled layout** | Moderate: needs a layout whose 16-byte row alignment is provable, and a host model for it. The 16-byte-row invariant is the binding constraint (`crates/cuda-device/src/shared.rs:83`, generated docstrings). | Execute: yes. |
| 5 | **x4 / normal pointer vs shared-u32 bit-for-bit equivalence** (`ADDRESS_ONLY` A/B, G1) | Moderate: two kernels that differ only in the address expression. The lowering difference is already pinned at IR level by `crates/mir-lower/tests/lowering_test/matrix_memory.rs:831-911` (NVPTX: one `inttoptr`; libNVVM: raw operand, no round-trip), so this is a device confirmation, not a discovery. | Execute: yes; **also requires an SM80 A/B timing run for the G1 verdict.** |
| 6 | **`mma_m16n8k16_f32_bf16` consuming `ldmatrix` fragments** | High: no example exercises this pairing at all (§1). Catalog-mandated floor is `sm_80` / PTX 7.0 (`intrinsics/catalog.json:105511` entry; `target.minimum_ptx = "7.0"` at `:105605`, `minimum_sm 80` at `:105610-105611`, `targets = "all"` at `:105616`), so **only this one is genuinely an `sm_80`-and-above feature**; the ldmatrix rows above are all `sm_75`+ features. | Execute: yes. |
| 7 | **SM80 compile evidence** for any row | Structural, not device: needs a checkout with the pinned nightly + CUDA 13.3 toolkit, then `cargo oxide build generated_ldmatrix --arch sm_80` (direct-LLVM PTX, ptxas-verifiable) and `cargo oxide emit-ltoir generated_ldmatrix --arch sm_80` (libNVVM LTOIR). No GPU required for either. | **No.** |
| 8 | **real A100 run** for any row | Needs the A100 host. Commands are pre-written in the task (§4, `cargo oxide run/sanitize generated_ldmatrix --arch sm_80`). | Yes, unavoidably. |

**What can be filled without a real SM80 GPU:** every backend-availability cell (already `SUPPORTED` from catalog + evidence records), the whole `SM80 compile` column (rank 7 — `build`/`emit-ltoir --arch sm_80` needs only the toolchain, and the `sm_80` gate is already satisfied by `crates/cuda-oxide-codegen/src/generated_intrinsic_targets/ldmatrix.rs:33-58`), all host-side expected-value models and negative controls, and `cargo oxide list --json` / `--emit-nvvm-ir` / PTX inspection. **What cannot:** the `real A100 run` column, the A/B timing half of G1, and memcheck/synccheck.

**Single biggest uncovered cell.** Not a shape/transpose/address combination but the **pointer-vs-shared-u32 equivalence for the same shape, executed on the same hardware** — and specifically its A/B form. The reason it dominates: ten of the twelve combinations are merely *unexecuted variants of the same instruction with the same documented contract*, and filling them is mechanical (rank 1–2). The address-path question that motivates issue #1235 (G1) is the only one where the tree contains **no** device-level comparison at all: `generated_ldmatrix` runs the shared-u32 form in one kernel (`main.rs:305-336`) and the pointer form in two *different* kernels with different indexing (`:344-375`, `:383-415`), and never compares the two fragment sets against each other. So a passing `generated_ldmatrix` cannot currently be cited as evidence that the two address forms are interchangeable — only that each independently matches the host model.

**If forced to name one shape/transpose/address cell as the biggest gap:** `x4 / trans / shared-u32` (row 12). It is the transposed shared-u32 case with the most lane addresses (all 32, no replication escape hatch), it is a call away from existing coverage (`main.rs:243`), and it is the combination most likely to expose a lane/row mapping error in the `.shared::cta.u32` path — yet it has never been executed or checked.

---

## 7. Could not determine (explicit)

| item | status | reason |
|---|---|---|
| SM80 compile of any of the 12 combinations | `UNVERIFIED` | gate admits `sm_80`, but no `sm_80` artefact exists in-tree and this host cannot compile (header). CI compiles the example at `sm_90` only (`scripts/smoketest.sh:395-399`, `:2026-2033`). |
| real A100 run of any of the 12 combinations | `NOT_RUN` | no GPU on this host; no CI lane executes examples. |
| Whether `legacy_ldmatrix_compile_oracle` survives into the emitted PTX | not inspected to a conclusion | kernel collection scans all CGUs for the reserved kernel prefix (`crates/rustc-codegen-cuda/src/collector.rs:25-33`, `:408`, `:490`), which implies all `#[kernel]` fns are exported regardless of host use — but no artifact was produced here to confirm it. This does **not** change the audit conclusion: the kernel is never launched (`main.rs:279-424`), so ten combinations have no runtime evidence either way. |
| Whether `cargo oxide list --json` currently lists `generated_ldmatrix` | not executed | no Rust toolchain on this host. The listing code path is fully read (`crates/cargo-oxide/src/commands/examples_list.rs:56-146`) and the directory satisfies every listed requirement. |
| `#1237`'s final discussion content | out of scope | the task fixes it as upstream-established ("CLOSED as a duplicate; did not deliver a carrier implementation"); no network fetch was performed. |
