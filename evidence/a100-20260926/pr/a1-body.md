## Why

#399 asked for small fixed-size local arrays consumed through iterator adapters
to stay in SSA. The maintainer's build on current main (the only comment on the
issue) shows the picture has flipped since the report:

```
iterator adapters (the issue)      -> 0 ld/st.local, 50 PTX instructions
index loop with a break            -> 4 st.local + 1 ld.local, 62 PTX instructions
```

Nothing in the tree asserts either half. `copy_aggregate_borrow` is the only
small-aggregate no-`.local` check and it is #400's shape — struct by value,
`%`-bounded index, no iterator. So a change that put the iterator form back into
local memory would not fail anything.

## What

A two-kernel example plus the code-shape check between them:

- `iterator_consumed_array` — the reported shape: a `[f32; 4]` read through
  `iter().copied().take(n.min(4)).enumerate()` with `n` taken from a runtime
  buffer, so the bound is not a constant the pipeline can fold away.
- `indexed_array_control` — the same array and the same arithmetic behind
  `for k in 0..4 { if k >= n { break } }`. This one is expected to keep the
  array in local memory.
- `verify-code-shape.sh` asserts the first kernel has no `.local`, `ld.local` or
  `st.local`, **and** that the second still does. A check that only asserted the
  absence would pass on a script that looked at nothing.
- The host oracle covers every bound the adapters can see — 0, 1, a middle
  value, exactly the array, and past its end — so `take` clamping is exercised on
  the device rather than assumed.

## Verified

A100-SXM4-40GB, `--arch sm_80`:

| item | result |
|---|---|
| `cargo oxide run` | both kernels match the host on all 128 lanes |
| `verify-code-shape.sh` | PASS |
| `sanitize --tool memcheck` | 0 errors |

What the PTX actually shows:

```text
iterator_consumed_array    (no .local, no ld.local, no st.local)
indexed_array_control      .local .align 4 .b8 __local_depot0[16];
                           st.local.b32 [%rd18],    1082130432;
                           st.local.b32 [%rd18+4],  -1069547520;
                           st.local.b32 [%rd18+8],  1068149419;
                           st.local.b32 [%rd18+12], -1098907648;
                           ld.local.b32 %r10, [%rd18];
```

That is the maintainer's `4 st.local + 1 ld.local` reproduced exactly, which is
what makes the negative control meaningful.

`scripts/sync-example-locks.sh --check` passes (238 example locks agree), and
`scripts/check-example-smoketest-contract.sh` accepts the new success marker.

## Scope

Test-only. No compiler change, and no claim that the iterator form is broken —
it is not. Reading the code, the pass behind this
(`scalarize_borrowed_aggregate_reads`) is also no longer opt-in: it is driven by
`promote_and_unroll` in `cuda-oxide-codegen/src/prep.rs`, and the
`CUDA_OXIDE_MIR_PASSES` registry contains one unrelated entry, so the issue's
`optional-opts` label no longer describes main. Fixing that label is not this PR.

Related to #399 (#399 is closed; this does not reopen it).

## Limitations

- One A100 SKU, one target (`sm_80`). `sm_75` and the libNVVM route were not run.
- The shape check reads PTX. It says nothing about ptxas register allocation or
  spills; local memory in the PTX and local memory in the cubin are not the same
  claim, and this only asserts the former.
- The array is four elements. Sizes at and past the pass's candidate budget were
  not probed, so this is a regression guard for the reported shape, not a survey
  of where scalarization stops.
