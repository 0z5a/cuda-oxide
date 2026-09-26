# A1 · #399 — result

**Status: `ALREADY_FIXED_NO_TEST`. Delivered as the permitted test-only
follow-up; no implementation, and #399 is not reopened.**

Full audit: [`closure-audit.md`](closure-audit.md) (260 lines, read-only,
frozen at `ec4aa4797956534578a1af010f86252a0b6d8626`).
Draft PR: [#1340](https://github.com/NVlabs/cuda-oxide/pull/1340).

## Decision

| question | answer |
|---|---|
| closure provenance | **`CLOSURE_COMMIT: UNKNOWN`** — 0 hits for `399` across all 1276 reachable commits (subjects *and* bodies) and every tracked file; the close window 2026-09-01..09-03 contains no scalarization commit |
| does main handle the reported iterator form? | **yes** — the maintainer's build on main reports 0 `ld/st.local` for the iterator form and 4 `st.local` + 1 `ld.local` for the index-loop form |
| is there a regression test? | **no** — hence `ALREADY_FIXED_NO_TEST` |

The pass behind this is also **default-on, not opt-in**: it is driven by
`promote_and_unroll` (`cuda-oxide-codegen/src/prep.rs:42-49`, `pipeline.rs:258`)
and the `CUDA_OXIDE_MIR_PASSES` registry holds exactly one unrelated entry
(`warp-aggregate-constant-fp-atomics`). The issue's `optional-opts` label no
longer describes main.

## What was delivered

`crates/rustc-codegen-cuda/examples/iterator_local_array_regression` — a
two-kernel example and the code-shape check between them.

| kernel | expected | observed on A100 / sm_80 |
|---|---|---|
| `iterator_consumed_array` (`iter().copied().take(n.min(4)).enumerate()`) | no local memory | no `.local`, `ld.local`, `st.local` |
| `indexed_array_control` (`for k in 0..4 { if k >= n { break } }`) | keeps local memory | `.local __local_depot0[16]`, 4× `st.local.b32`, 1× `ld.local.b32` |

`verify-code-shape.sh` fails if **either** half breaks. The control reproduces
the maintainer's exact counts, which is what gives the assertion teeth.

Device runs: both kernels match the host oracle on all 128 lanes; `memcheck`
reports 0 errors. `sync-example-locks.sh --check` and
`check-example-smoketest-contract.sh` both pass.

## Residual gaps found (not acted on)

The audit documents seven patterns the existing pass does not handle, with code
proof. The strongest is **G1**: a bare local array `let arr: [T; N]` whose
address is taken — exactly what `arr.iter()` produces — is never considered,
because both phases hard-require a struct aggregate
(`scalarize_borrowed_aggregate_reads.rs:172` and `:516`). Also: the aggregate
pointer must be an entry-block argument of an `alwaysinline` helper (`:494-521`);
the index must be `urem`- or exact-`assert`-guarded (`:559-612`); a 16-candidate
budget (`dialect-mir/src/ops/aggregate.rs:1185`); mutation disqualifies the slot
(`:186-201`).

**None of these is the issue's own shape**, which main already handles. Acting
on them would be a new scope and needs maintainer confirmation first, so they
are reported rather than implemented.

## Limitations

- One A100 SKU, one target (`sm_80`); `sm_75` and the libNVVM route are `NOT_RUN`.
- The shape check reads PTX. Local memory in PTX is not the same claim as local
  memory in the cubin; ptxas register allocation and spills were not inspected.
- The array is four elements, so this guards the reported shape rather than
  surveying where scalarization stops.
- `CLOSURE_UNKNOWN` is recorded as a fact, not resolved: an unfetched branch
  close cannot be excluded from a local clone.
