## Why

#399 asked for small fixed-size local arrays consumed through iterator adapters
to stay in SSA. The maintainer's build on current main (the only comment on the
issue) shows the picture is mixed:

```
iterator adapters (the issue)      -> 0 ld/st.local, 50 PTX instructions
index loop with a break            -> 4 st.local + 1 ld.local, 62 PTX instructions
```

Nothing in the tree asserts either half. `copy_aggregate_borrow` is the only
small-aggregate no-`.local` check and it is #400's shape — struct by value,
`%`-bounded index, no iterator. A change that put the iterator form back into
local memory would not fail anything.

## What the revision does

**Runtime coverage.** `iterator_consumed_array` is the reported shape: a
`[f32; 4]` read through `iter().copied().take(n.min(LEVELS)).enumerate()` with
`n` from a runtime buffer, so the bound is not a constant the pipeline can fold.
`indexed_array_control` computes the same value through the loop form and is a
semantic comparison, not a performance contract.

The bounds now cover no iteration, the last element, exactly the array, values
past its end and `u32::MAX`, and the host asserts that coverage so a later
change to the generator cannot quietly drop it. Because the expression clamps
before `take`, the past-the-end entries exercise the runtime-bound expression as
a whole rather than `Iterator::take` on its own — the description says so and
the code does not change shape to make a stronger sentence true.

Each arm starts from a NaN-filled output buffer. A lane the kernel never writes
stays NaN and cannot be read back as a value a previous arm left behind, and the
verifier reports the first offending lane with its bound, expected and actual
value.

**The PTX checker.** `verify-code-shape.sh` resolves the entry by name at the
start of a `.visible .entry` line and inspects only that body, so a similar
name, a mention in a comment, a truncated body or an unrelated kernel's local
memory cannot decide the result. `indexed_array_control`'s local usage is
**reported, not asserted** — the previous revision required it to keep spilling,
which is a performance contract this test has no business enforcing.

The parser carries ten fixtures, run in the no-argument mode the repository
smoke test already calls, before the real PTX is read: the clean shapes, a
`.local` declaration and a local load in the target kernel, a missing control, a
name that is only a prefix, a name that appears only in a comment, a truncated
body, an empty file, and local memory in an unrelated kernel. An explicit path
argument skips the self-test and checks only that file.

No new workflow, runner hook or CI entry point: the existing
`scripts/smoketest.sh` hook that calls the script by path is unchanged.

## Verified

A100-SXM4-40GB, `--arch sm_80`, rustc `nightly-2026-08-28`:

| item | result |
|---|---|
| `cargo oxide run iterator_local_array_regression` | both kernels match the host on all 128 lanes |
| `verify-code-shape.sh` (no argument: self-test + real PTX) | PASS, 10/10 fixtures |
| `verify-code-shape.sh <ptx>` | PASS (`indexed_array_control: uses local memory`) |
| `scripts/smoketest.sh --compile-only --only '^iterator_local_array_regression$'` | PASS 1/1 |
| `compute-sanitizer --tool memcheck` | 0 errors |

PTX, per entry:

```text
iterator_consumed_array    (no .local, no ld.local, no st.local)
indexed_array_control      .local .align 4 .b8 __local_depot0[16];
                           st.local.b32 [%rd18],    1082130432;
                           st.local.b32 [%rd18+4],  -1069547520;
                           st.local.b32 [%rd18+8],  1068149419;
                           st.local.b32 [%rd18+12], -1098907648;
                           ld.local.b32 %r10, [%rd18];
```

### Fault injection

Each mutation was applied in a scratch copy, run, and reverted; the file hashes
were checked before and after (`verify-code-shape.sh` `c447e2d9…`, `main.rs`
`b918c9b2…`).

| mutation | observed |
|---|---|
| checker stops testing for local memory | fixture suite fails, checker exits 1 |
| `indexed_array_control` skips one lane's write | runtime verifier fails: `lane 11 (bound 3) was never written` |
| checker deliberately fails | `smoketest.sh` reports `FAIL (exit=1)` |

After restoring, the run and both checker modes pass again.

## Scope

Test-only. No compiler change, and no claim that the iterator form is broken —
it is not. The pass behind this is also no longer opt-in: it is driven by
`promote_and_unroll` (`cuda-oxide-codegen/src/prep.rs:42-49`), and the
`CUDA_OXIDE_MIR_PASSES` registry holds one unrelated entry, so the issue's
`optional-opts` label no longer describes main. Relabelling the issue is not
this PR.

Related to #399 (#399 is closed; this does not reopen it).

## Limitations

- One A100 SKU, one target (`sm_80`). `sm_75` and the libNVVM route were not run.
- The shape check reads PTX. It says nothing about ptxas register allocation or
  spills: local memory in the PTX and local memory in the cubin are not the same
  claim, and only the former is asserted.
- The array is four elements, so this guards the reported shape rather than
  surveying where scalarization stops.
- The fixtures are minimal hand-written fragments, not captured compiler output.
