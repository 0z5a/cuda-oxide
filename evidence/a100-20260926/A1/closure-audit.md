# A1 · #399 Closure and Coverage Audit (P0)

**Audit date:** 2026-09-26 · **Mode:** read-only (no repo file modified, no write to GitHub)

## 0. Frozen base

| Field | Value |
|---|---|
| Checkout | `/Users/0z5a/Documents/infra/cuda-oxide-a100-20260926/oxide-mac` |
| `git rev-parse HEAD` | `ec4aa4797956534578a1af010f86252a0b6d8626` |
| HEAD subject | `Merge pull request #1327 from uurl/fix/packed-as3-local-storage-contract` (2026-09-24) |
| Working tree | **clean at audit start** (`git status --short` empty). A concurrent writer later left `crates/mir-lower/src/convert/ops/arithmetic.rs` and `crates/mir-lower/tests/lowering_test/{main,shift_representation}.rs` dirty (shift-representation work, unrelated to A1). **Every file cited by this audit was re-checked and is unmodified**; this audit wrote no repository file. |
| History shape | 1276 commits reachable, 186 merges, **not** shallow, dates 2026-04-24 … 2026-09-24 |
| Refs mirrored locally | 17 total (`refs/heads/main`, `refs/heads/pr-1304`, `refs/heads/pr-1305`, 3 tags, 11 `refs/remotes/origin/*` incl. `origin/HEAD` ⇒ 10 `origin` branches) |
| Pinned toolchain | `nightly-2026-08-28` (`rust-toolchain.toml:2`) — matches the maintainer's LLVM 23 build in §5 |

All commands in this report were `git log/show/grep/rev-parse`, `grep`, `read`, `ls`. No build, no test run, no network fetch, no state-changing git command.

---

## 1. Closure provenance

### 1.1 Searches performed and their results

| Command | Result |
|---|---|
| `git log --all --oneline -i --grep='399'` | **0 hits** |
| `git log --all --format='%H'` + body-scan for `399` over all 1276 commits | **0 hits** |
| `git log --all --oneline -i --grep='iterator'` | 13 hits; none names #399 (top hits: `1f6972f2` #1203, `fe616639` #1201, `5847eee7` #1157, `2272e69e` #832, `ffe1135c` #758) |
| `git log --all --oneline -i --grep='scalariz'` | 9 hits; the only aggregate-scalarization commit is `98b79955` (#650) |
| `git log --all --oneline -i --grep='local array'` | 1 hit: `83d23ac3` (#694), unrelated (enum-array constants) |
| `git log --all -i --grep='#398'` / `--grep='#400'` | both hit only `98b79955` (#650) |
| `git grep -n -I -E '\b399\b'` over tracked files | **0 hits** |
| Commit bodies 2026-08-25…2026-09-24 containing `scalariz\|iterator\|local array\|closes\|fixes` | only `#1292`, `#1277`, `#1290`, `#1211`, `#1218`, `#1180` — none is #399 |

### 1.2 Commits in the close window (2026-09-01 … 2026-09-03)

| SHA | Date | Author | Subject |
|---|---|---|---|
| `8ac82ec7` | 2026-09-04 | nihalpasham | `cargo-oxide: build the backend from the project's cuda-oxide dependency commit (#1233)` |
| `cf18cfcc` | 2026-09-04 | nihalpasham | `rustc-codegen-cuda: collect std's float wrappers instead of rejecting them when MIR inlining is off (#1230)` |
| `28e8953c` | 2026-09-03 | Melih Elibol | `Use the shared cutile-rs host crates in cuda-oxide (#1102)` |
| `596a6353` | 2026-09-03 | nihalpasham | `cuda-macros: keep the kernel specialization alive with black_box, not a null-pointer volatile pair (#1226)` |
| `0b055635` | 2026-09-03 | Raul Estrada | `ptx-schedule: classify grid dependency control sites (#1214)` |
| `d843d5b8` | 2026-09-03 | Ziming Wang | `llvm-export: encode enum debug tags as unsigned storage (#1222)` |
| `332fe6dc` | 2026-09-03 | Raul Estrada | `ptx-schedule: classify tensor-map mutations as schedule sites (#1215)` |
| `a1b4f118` | 2026-09-02 | nihalpasham | `debug: full device debug keeps MIR optimizations, disables only the two debugger-hostile passes (#1221)` |
| `073b5bd9` | 2026-09-02 | nihalpasham | `debug: fix LLVM 23 silently dropping device debug info for kernel-local statics (#1216)` |
| `f2bcaa53` / `36831410` / `369cf214` / `c0a6d631` / `fc3ec002` | 2026-09-02 | nihalpasham | lockfile / vscode / docs / trybuild chores |

None of these touches aggregate scalarization or iterators. The commit that *does* is outside the close window and references different issues.

### 1.3 The scalarization commit that does exist

| Field | Value |
|---|---|
| SHA | `98b79955e7ce8bf1b01727fbe6567da756b09d3e` |
| Subject | `codegen: scalarize bounded borrowed aggregate reads in MIR (#650)` |
| Author | Raul Estrada `<2509572+uurl@users.noreply.github.com>` |
| AuthorDate | 2026-08-05 07:00:09 -0600 |
| CommitDate | 2026-08-05 18:30:09 +0530 (GitHub merge) |
| Co-author | `nihalpasham <nihalp@nvidia.com>` |
| Issues referenced in its body | **#400** and **#398** (the post-mem2reg fail-closed gate) — **not #399** |
| Ancestry | `git merge-base --is-ancestor 98b79955 HEAD` → **true** |
| Files | `crates/mir-transforms/src/scalarize_borrowed_aggregate_reads.rs` (+1564), `crates/mir-transforms/src/lib.rs` (+1), `crates/cuda-oxide-codegen/src/{pipeline,prep}.rs`, `crates/dialect-mir/src/ops/aggregate.rs`, `crates/mir-lower/src/convert/ops/aggregate.rs`, and the new `examples/copy_aggregate_borrow/` (incl. `verify-code-shape.sh`) |
| Later edits to the pass (not reverts) | `1f0b065a` (#1186, 2026-08-28, typed pointer provenance), `b9847e95` (#1293, 2026-09-18, preserve assertions during unrolling) |

### 1.4 Verdict

> **`CLOSURE_COMMIT: UNKNOWN`**

No commit, merge message, or tracked file reachable from any of the 17 local refs references #399. The only implementation-level scalarization work in this history is `98b79955` (#650), and its own body attributes the fail-closed rule to #400/#398, not #399.

**Caveat (does not change the verdict):** the clone mirrors only 10 `origin/*` branches. A closing commit on a branch that was never fetched, or a close performed purely as a GitHub issue-event (no commit), cannot be excluded from this checkout. The upstream close reason (maintainer comment, 2026-09-03) describes a **verification on main**, not an iterator-specific code change, which is consistent with there being no attributable closing commit.

---

## 2. The existing pass

### 2.1 Location and registration

| Item | Evidence |
|---|---|
| Module | `crates/mir-transforms/src/scalarize_borrowed_aggregate_reads.rs` (1596 lines) |
| Exported | `crates/mir-transforms/src/lib.rs:18` — `pub mod scalarize_borrowed_aggregate_reads;` |
| Phase A entry | `scalarize_borrowed_aggregate_reads.rs:128` `canonicalize_read_only_aggregate_arguments` |
| Phase B entry | `scalarize_borrowed_aggregate_reads.rs:396` `canonicalize_bounded_borrowed_pointer_arguments` |
| Phase A call site | `crates/cuda-oxide-codegen/src/prep.rs:85-89` (after compiler-result forwarding, **before** `mem2reg`) |
| Phase B call site | `crates/cuda-oxide-codegen/src/prep.rs:140-141` (after post-mem2reg result forwarding, before `unroll_annotated_loops`) |
| Pipeline stage | `crates/cuda-oxide-codegen/src/pipeline.rs:276-284`, inside `prepare_mir_module` |

### 2.2 Opt-in or default-on? (the actual switch name)

**The pass is default-ON. It is NOT in the optional-pass registry.**

| Question | Answer | Evidence |
|---|---|---|
| Is there an optional-pass registry? | Yes | `crates/cuda-oxide-codegen/src/mir_pass_registry.rs:70-154` |
| Its switch name | **`CUDA_OXIDE_MIR_PASSES`** (comma-separated staged pass names) | `mir_pass_registry.rs:6` (`//! Optional MIR passes selected by CUDA_OXIDE_MIR_PASSES.`); read at `crates/cuda-oxide-codegen/src/options.rs:98-103` and `:173` |
| Passes registered in it at HEAD | **exactly one**: `"warp-aggregate-constant-fp-atomics"` (`PostMem2Reg`) | `mir_pass_registry.rs:146-154`; asserted by `mir_pass_registry.rs:256-270` |
| Is `scalarize_borrowed_aggregate_reads` registered? | **No** — the pass name does not appear anywhere in `mir_pass_registry.rs` | `grep -n 'scalariz' crates/cuda-oxide-codegen/src/mir_pass_registry.rs` → no match |
| What actually gates it | `if !preparation.promote_and_unroll { … return Ok(()); }` | `crates/cuda-oxide-codegen/src/prep.rs:42-49` |
| Value of `promote_and_unroll` | `!request.debug_kind.variables_enabled()` → **true** for every optimized build; false only under *full variable* debug info | `crates/cuda-oxide-codegen/src/pipeline.rs:258`; `crates/llvm-export/src/export/config.rs:59` |
| `CUDA_OXIDE_VERBOSE` effect | progress notes only; the pass reads no env | `prep.rs:19-23`; `scalarize_borrowed_aggregate_reads.rs:126-127`, `:149-151`, `:394-395`, `:433-438` |

So the issue's `optional-opts` label does **not** describe this pass at HEAD. The only env knob that changes its behaviour at all is full-variable-debug mode, which disables the entire `promote_and_unroll` preparation stage (`pipeline.rs:258-275`).

### 2.3 What it recognizes (precise)

**Phase A** — `analyze_alloca` (`scalarize_borrowed_aggregate_reads.rs:169-303`), pre-mem2reg:

- entry-block `mir.alloca` whose pointee is a **`MirStructType`** — `:172`;
- initialized **exactly once** by a `mir.store` of an **entry-block argument** of the same type — `:186-201`;
- every other use of the slot is a `mir.field_addr` (`:203`) belonging to the exact read-only graph validated by `analyze_field_path` (`:220-303`): constant field projections down to `mir.array_element_addr` + non-volatile `mir.load`;
- rewrite (`:311-360`): `mir.extract_field` + `mir.extract_array_element` replacing the pointer chain.

**Phase B** — `analyze_borrowed_pointer_read` (`:441-557`), post-mem2reg:

- the scan target is a `MirArrayElementAddrOp` (`:416-418`, `:447`);
- its base `mir.field_addr` must be in the **same block**, immutable, with **exactly one use** which is that `array_element_addr` at operand 0 (`:452-461`);
- the array field's pointee is a `MirArrayType` of size > 0 (`:469-474`);
- the element pointer is immutable with exactly one use, consumed by a non-volatile `mir.load` (`:476-492`);
- the aggregate pointer must be an **entry-block argument** (`:494-521`) of a `MirFuncOp` carrying the **`alwaysinline`** attribute (`:501-507`), immutable (`:512-514`), with a `MirStructType` pointee (`:515-516`);
- **caller provenance**: every `mir.call` of that helper must pass a pointer traceable through `MirCastKindAttr::PtrToPtr` reborrows to a caller-local `mir.alloca` of exactly the helper's aggregate type — `all_call_sites_pass_owned_aggregate` (`:624-651`), `pointer_is_owned_aggregate_slot` (`:662-686`); reserved device symbols and helpers with no visible call site are rejected (`:635-640`);
- **bounded index** (`bounded_pointer_index`, `:559-612`): either (a) the index is already `mir.rem(value, constant)` (`:572-579`), or (b) the load block has **exactly one predecessor** which ends in `mir.goto`, whose immediately preceding op is `mir.assert(mir.lt(index, constant))` with the same index and same-width bound (`:581-611`);
- the candidate count must satisfy `0 < C ≤ array_size` and `C ≤ MAX_SCALARIZED_CANDIDATES` (`validate_candidate_count`, `:698-703`);
- rewrite (`:705-758`): loads the whole array field at the original access point, materialises the equivalent `mir.rem` for the asserted case, and replaces the load with `mir.extract_array_element`.

**Profitability is not decided here.** The pass canonicalizes "independently of the runtime index shape" (`:30-34`); the actual `extractvalue`+select expansion vs. memory fallback lives in the LLVM lowering: `crates/mir-lower/src/convert/ops/aggregate/array_extract.rs:24-34`, `:57-67`, `:109-120`. The cap is shared: `MAX_SCALARIZED_CANDIDATES = 16` at `crates/dialect-mir/src/ops/aggregate.rs:1185`.

### 2.4 What it rejects (fail-closed)

| Rejection | Code | Test |
|---|---|---|
| non-struct pointee (bare array/tuple/slice aggregate) | `:172`, `:516` | — (structural guard) |
| additional store, volatile store, store of a non-entry-argument | `:186-201` | `:1037`, `:1048` |
| store through a derived pointer | `analyze_field_path` | `:1059` |
| non-`PtrToPtr` casts, block arguments, shared/global, phi/select, offsets | `:662-686` | `:1526` |
| device-export symbol, helper with no visible call site, call site passing a pointer parameter | `:635-651` | `:1560`, `:1577`, `:1543` |
| non-`alwaysinline` helper | `:501-507` | `:1474` |
| mutable or multi-use field/element pointer | `:454`, `:466`, `:480`, `:485-487` | — |
| volatile load | `:490` | `:1492` |
| index not `urem`- or `assert`-bounded, bound > array size, or bound > 16 | `:559-612`, `:698-703` | `:1009`, `:1023` |
| index not unsigned | `:565-570` | — |

---

## 3. Is the iterator form covered by an existing test?

### 3.1 `iter().copied().take(t).enumerate()` over a small local array — `NO_TEST_FOUND`

| Search | Result |
|---|---|
| `grep -rn '\.iter()\.copied()\|\.copied()\.enumerate()\|\.iter()\.take(\|\.iter()\.skip(' crates/*/tests crates/*/src crates/rustc-codegen-cuda/examples` | every hit is **host-side harness** code, e.g. `examples/inline_ptx/src/main.rs:182,235,256`, `examples/field_array_assign/src/main.rs:86`, `examples/slice_reslice/src/main.rs:72` — none is inside a `#[cuda_module]` kernel |
| `.take(` / `.skip(` in device regions | **zero** runtime-bounded uses; the only `.take(`-matching line inside a device region is `examples/enum_state_transitions/src/main.rs:34` (`Option::take`, unrelated) |
| Device-region scan for iterator patterns (44 hits) | the adapter chains present are `into_iter()`, `.enumerate()`, `.iter()`, `.sum()`, `.chain()` — never `.copied()`, `.take(t)`, or `.skip(s)` over a small array |

### 3.2 Regression tests that assert absence of `.local` / `ld.local` / `st.local` in PTX

| Path:line | What it asserts | Relation to #399 |
|---|---|---|
| `crates/rustc-codegen-cuda/examples/copy_aggregate_borrow/verify-code-shape.sh:33` | `grep -Eq '(^\|[[:space:]])\.local\|ld\.local\|st\.local'` over the `borrowed_copy_aggregate` entry body must fail | **Closest existing regression, but not the iterator form.** It covers `shape.counts[axis]` / `shape.periodic_bit(axis)` with `axis = linear_index % shape.counts.len()` — a struct-by-value with `[u32;3]`/`[bool;3]` fields read through `#[inline(always)]` methods (issue **#400**, see `examples/copy_aggregate_borrow/src/main.rs:7-11` doc, `:24-33` accessors, `:41-47` kernel). Auto-run by `scripts/smoketest.sh:2104-2130`. |
| `crates/cuda-oxide-codegen/tests/wgmma_pipelined_counted_loop_ptx.rs:389`, `:517` | no `.local`/`ld.local`/`st.local` in the counted pipeline | WGMMA descriptor-recurrence kernel; message "the production counted pipeline unexpectedly materializes local memory" |
| `crates/cuda-oxide-codegen/tests/wgmma_counted_pipeline_carrier_ptx.rs:621`, `:758` | same | WGMMA pipeline carrier |
| `crates/cuda-oxide-codegen/tests/wgmma_pipeline_carrier_ptx.rs:245` | same | WGMMA pipeline carrier |
| `crates/cuda-oxide-codegen/tests/wgmma_counted_loop_ptx.rs:387` | same | counted-loop carrier |
| `crates/cuda-oxide-codegen/tests/wgmma_value_carrier_ptx.rs:363` | same | WGMMA value carrier |
| `crates/llvm-export/tests/export_test/grid_constant.rs:313` | `!ptx.contains(".local")` — "grid parameters must not become thread-local copies" | `__grid_constant__` parameters, not a Rust local array |

None of these kernels consumes a local array through an iterator adapter.

### 3.3 Pass-level tests that exist (MIR only, no PTX)

- Phase A unit tests: `scalarize_borrowed_aggregate_reads.rs:991` (`bounded_rem_rewrites_large_array_with_small_candidate_set`), `:1009`, `:1023`, `:1037`, `:1048`, `:1059`.
- Phase B unit tests: `:1428` (`asserted_immutable_pointer_read_is_canonicalized_after_mem2reg`), `:1454`, `:1474`, `:1492`, `:1526`, `:1543`, `:1560`, `:1577`.
- Every fixture builds a **struct-typed** aggregate (`:800`, `:1258`), consistent with the `MirStructType` guard.
- Lowering unit tests: `crates/mir-lower/src/convert/ops/aggregate/array_extract.rs:292` (`bounded_urem_array_extract_stays_in_ssa`), `:307` (`unbounded_array_extract_keeps_memory_fallback`), `:314`, `:321`.

**These prove the pass and the lowering; they do not exercise an end-to-end device kernel with an iterator chain, and they make no PTX assertion.**

### 3.4 The repository states the gap itself

| Path:line | Text |
|---|---|
| `crates/rustc-codegen-cuda/examples/slice_iterators/src/main.rs:15-17` | "This regression intentionally stays on forward iterator paths. It does not exercise reverse/from-end iteration, dedicated MIR `Subslice` projection regressions, **local-array iterator scalarization**, or `as_chunks`." |
| `crates/rustc-codegen-cuda/examples/slice_iterators/README.md:16` | `- local-array iterator scalarization;` (listed under "It does not cover:") |

---

## 4. Does any example cover it?

Device-code examples that *do* consume a small fixed-size local array through an iterator adapter (runtime-verified only — **no** `.local` assertion in any of them):

| Path:line | Shape |
|---|---|
| `crates/rustc-codegen-cuda/examples/array_for_loop/src/main.rs:51-56` | `let arr: [u32; 4] = [t, t+1, …]; for x in arr { acc += x; }` (issue #138) |
| `crates/rustc-codegen-cuda/examples/array_for_loop/src/main.rs:66-75` | `let pts: [Point; 4] = […]; for p in pts { … }` |
| `crates/rustc-codegen-cuda/examples/array_for_loop/src/main.rs:87-89` | `array::from_fn(…)`, `.map(…)`, `mapped.into_iter().sum()` |
| `crates/rustc-codegen-cuda/examples/cast_tests/src/main.rs:357-361` | `let arr: [f32; 3] = […]; for val in arr.as_slice().iter() { … }` |
| `crates/rustc-codegen-cuda/examples/redux_f32/src/main.rs:34-49` | `let values = [<8 runtime warp results>]; for (slot, value) in values.into_iter().enumerate()` |
| `crates/rustc-codegen-cuda/examples/generated_intrinsics/src/main.rs:129`, `:135`, `:203`, `:209`, `:294`, `:300`, `:306`, `:375` | `values_*: [T; N]` local arrays, `.into_iter().enumerate()` |
| `crates/rustc-codegen-cuda/examples/generated_intrinsics_blackwell/src/main.rs:56`, `:82`, `:429` | same |
| `crates/rustc-codegen-cuda/examples/mma_mxf8f6f4/src/main.rs:63` | `let d = wmma::…(…)` → `[f32; 4]`, `.into_iter().enumerate()` |

Caveats:

- `redux_f32` is the only one whose array is genuinely **runtime-initialised** (eight warp intrinsic results, `:34-43`) and consumed by a runtime index adapter — but it is **Blackwell-only**: `examples/redux_f32/Cargo.toml:12` (`--arch=sm_100a`), `scripts/smoketest.sh:84` (`SM100_COMPILE_EXAMPLES=(redux_f32)`), `cuda-oxide-book/advanced/warp-level-programming.md:199`.
- No example anywhere uses `take(t)` / `skip(s)` / `.copied()` with a **runtime** bound over a local array. `array_for_loop`'s `for x in arr` is the const-length `array::IntoIter` path (docs `:8-14`), not a runtime-bounded adapter chain.

---

## 5. Decision

> ### `ALREADY_FIXED_NO_TEST`

**Justification, in the order the decision rule requires:**

1. **Main handles the iterator form.** The upstream maintainer's build on main (nightly-2026-08-28 / LLVM 23 — the same nightly pinned in this checkout) reports the iterator form `for (k, c) in coefficients.iter().copied().take(n.min(4)).enumerate()` at **0 `ld.local`/`st.local`, 50 PTX instructions**, correct on an RTX 5090 at both `sm_80` and `sm_120a`, with **no iterator-specific pass** involved. That is the "handled" half of the decision.
2. **No regression test exists.** §3.1 returns `NO_TEST_FOUND` for the adapter chain; §3.2 shows the only no-`.local` example assertion on a small aggregate is `copy_aggregate_borrow/verify-code-shape.sh:33` for issue **#400** (struct-by-value, `%`-bounded index, no iterator); §3.4 shows the repo itself lists "local-array iterator scalarization" as explicitly out of scope for its only device-side iterator regression.
3. **This is therefore the "only add tests" case**, not a re-implementation. Nothing in local history (`98b79955` #650) is an iterator implementation to duplicate or extend for this form.

**Separate, and equally required to be stated:**

> **`CLOSURE_COMMIT: UNKNOWN`** — see §1. The evidence *reason* is a maintainer verification, and no local commit, PR, or tracked file references #399. Nothing in this report attributes the closure to a specific change.

If the verdict were restricted to *provenance alone*, it would be `CLOSURE_UNKNOWN`; the overall decision above is `ALREADY_FIXED_NO_TEST` because the behavioural question ("does main handle the iterator form?") is answered by the upstream evidence while the provenance question is not.

**Residual work implied by this decision:** exactly one additive regression test — a device kernel that consumes a small fixed-size local array through a runtime-bounded iterator adapter and asserts absence of `.local`/`ld.local`/`st.local` in that kernel's PTX. No pass change, no default-policy change.

---

## 6. Residual-gap scan

Patterns in the source that `scalarize_borrowed_aggregate_reads.rs` **demonstrably does not handle**, with the code that proves it. All are *pass-scope* gaps; the issue's specific iterator shape is not among them (main handles it downstream of this pass).

| # | Pattern not handled | Proof | Consequence |
|---|---|---|---|
| G1 | **A bare local array — `let arr: [T; N]` — whose address is taken** (exactly what `arr.iter()`, `arr.as_slice()`, or `&arr` produces). Both phases are gated on the aggregate being a struct. | `:172` — `pointee.deref(ctx).downcast_ref::<MirStructType>()?;` (phase A); `:516` — `aggregate_type.deref(ctx).downcast_ref::<MirStructType>()?;` (phase B). Fixtures are struct-typed only (`:800`, `:1258`). | The pass never considers a bare-array alloca. In this checkout the pass is the only MIR-transforms producer of `extract_array_element` for pointer-shaped reads; `forward_compiler_result_bundles` cannot take over because it requires the importer's bundle marker ("Ordinary Rust aggregates are never candidates because only the importer can attach the marker", `crates/mir-transforms/src/forward_compiler_result_bundles.rs:16-17, :30`). |
| G2 | **A local array borrowed inside the kernel body itself** (not a by-value struct parameter). Phase B requires the aggregate pointer to be an entry-block *argument* of an `alwaysinline` helper. | `:494-499` (`entry_block = aggregate_pointer.defining_block()`, must be the region's first block), `:518-521` (must be at a position in `entry_block.arguments()`), `:503-507` (function must carry `alwaysinline`). | An in-kernel `&arr` chain is structurally excluded even when its index is provably bounded. |
| G3 | **Runtime index without a `urem` or an exact `assert(index < C)` guard.** | `bounded_pointer_index` `:572-579` (only a `MirRemOp` divisor) and `:581-611` (exactly one predecessor, terminator `mir.goto`, immediately preceding `mir.assert`, condition defined by `MirLtOp` in that block with operand 0 == index). | An index that is merely loop-bounded (`for i in 0..4`) or a `select`/`min`-clamped value is rejected. `crates/rustc-codegen-cuda/examples/array_index/src/main.rs:82-96` documents exactly this shape: `test_runtime_index_read` "Expected: PASS (but inefficient - copies whole array per read)" (`:83`) with the file-level table at `:18-21` calling it `✓ PASS (alloca+gep+load)` (`:19`) — and its module doc `:79` names the lowering `ProjectionElem::Index → MirExtractArrayElementOp → alloca+gep+load`. |
| G4 | **Candidate count above the shared cap of 16.** | `:698-703` (`candidate_count <= MAX_SCALARIZED_CANDIDATES`); cap at `crates/dialect-mir/src/ops/aggregate.rs:1185` (`pub const MAX_SCALARIZED_CANDIDATES: u64 = 16;`); lowering fallback at `crates/mir-lower/src/convert/ops/aggregate/array_extract.rs:57-67` with unit test `:314` (`oversized_urem_array_extract_keeps_memory_fallback`). | `[T; 32]` reached through any dynamic index keeps its memory form even with a proven bound. |
| G5 | **Any write to the aggregate** — mutation disqualifies the whole slot, and there is no re-materialisation path. | `analyze_alloca` `:186-201` (a second store, a volatile store, or a stored value that is not an entry-block argument returns `None`); tests `:1037` (`additional_store_rejects_the_entire_slot`), `:1059` (`store_through_derived_pointer_rejects_the_entire_slot`). | Read-modify-write on a local array (`examples/array_index/src/main.rs:252-263`) is out of scope; `examples/array_index/src/main.rs:16-21` marks runtime-index **writes** as `✗ FAIL (not implemented)`. |
| G6 | **Two reads of the same element address / any second use of the element or field pointer.** | `:454` (`field_pointer.num_uses(ctx) != 1` → `None`), `:466-468` (field pointer must be immutable), `:480` (`element_pointer.num_uses(ctx) != 1` → `None`), `:485-487`. | Multi-consumer read patterns are not canonicalized. |
| G7 | **Nested / non-array aggregate leaves.** | `:469-474` requires the field pointee to be a `MirArrayType`; there is no recursive descent into `[[T; M]; N]` or into array-of-struct fields. | Only one array level per field projection is reachable. |

**Not a gap (the issue's own shape):** `for (k, c) in coefficients.iter().copied().take(n.min(4)).enumerate()` — main produces register-only code for it (upstream maintainer build, §5 item 1), so it is deliberately **not** listed above.

**Strongest single residual:** **G1**, because it is the exact noun of the issue title ("small **local arrays** consumed through iterator adapters") and it is a hard type guard, not a heuristic miss — the pass cannot see a bare-array alloca under any provable bound.

---

## 7. Status of this audit

| Item | Status |
|---|---|
| P0 closure/coverage audit | **DONE** (this file) |
| Local reproduction / PTX measurement | `NOT_RUN` — no build, no GPU, no test execution was performed in this read-only audit; §5 item 1 relies on the upstream maintainer measurement supplied with the task |
| A100 / SM80 device verification | `NOT_RUN` |
| Pass modification | **Not performed and not implied** — the decision is test-only |
| Repo mutation by this audit | **None.** Only `evidence/A1/closure-audit.md` was written. Uncommitted changes visible in the checkout belong to a concurrent writer (§0). |
