## What

#1305 chooses the reduction form from the target floor at compile time, so the
two forms cannot be compared inside one binary — the comparison has to be two
builds of one source. That harness did not exist, and #1305 says so: *"no new
speedup measurement was made."*

Four arms, two per form:

- `api_chain` / `api_independent` drive `warp_reduce::<u32, _>`. One is a
  dependency chain, the other runs four unrelated reductions per iteration.
- `shuffle_chain` / `shuffle_independent` spell the butterfly out with
  `WarpCollective::shfl_xor`, so on one target the pair isolates the reduction
  from the rest of the build: same compiler, same binary.

## Correctness

The oracle is a **per-warp host simulation of the real data flow**, not a
comparison between the two arms — two arms can share a fault. Every reduction
reads the previous round's whole warp and only then broadcasts, so no lane
observes a value another lane already updated.

| property | how |
|---|---|
| both forms checked against the oracle | not against each other |
| four values per lane | the independent arms publish `a, b, c, d` instead of folding them into one XOR, so a fault in one operation is locatable |
| per-arm fresh output | each arm starts from the bitwise inverse of its oracle; a lane the kernel never writes cannot read back as a correct answer |
| iteration counts | 0, 1, 2, 3, 8, 17 — the degenerate ends and past the point where the sum recurrence reaches its fixed point |
| inputs | zero, `u32::MAX`, the top bit, `i32::MAX`, one, fixed-seed values, cycled so different warps differ |
| failure report | first mismatch with warp, lane, field, expected and actual |

## Metric definitions

The number is an **amortised collective cost**, not instruction latency. It is
timed with a host wall clock around launch plus stream synchronisation, so
launch and host overhead are small per reduction but not absent, and one warp
reduction is one collective — not one instruction, and not 32 results.

```
timer_kind=host_wall_clock_kernel_batch iterations=40000 grid_blocks=1 warps_per_block=4
amortized_ns_per_warp_reduction = kernel_ms * 1e6 / (iterations * blocks * warps * R)
```

`R` is 1 for the chain arms and 4 for the independent ones and is printed per
row. Arms are sampled round-robin inside one pass, so a clock or power drift
lands on all of them. No threshold is asserted: the harness reports, it does not
gate.

A scaling check runs the chain arms at four iteration counts and warns when time
does not grow with the count — what a loop the compiler removed would look like.

## Measured

A100-SXM4-40GB (cc 8.0, driver 595.91.07, no MIG), rustc `nightly-2026-08-28`,
`llc` LLVM `23.1.0-rust-1.100.0-nightly`, CUDA `13.0.88`.

**Test tree 1 — `main` alone** (`ec4aa479`), no fast-path implementation:
correctness passes, and `api_chain` reports 27.19 ns, the same as the
butterfly. The harness measures the API as the fallback it actually compiled
to; it does not label the run `redux` because the target is `sm_80`.

**Test tree 2 — #1305's dependency tree** (`cf4c55c3`) plus this harness,
`--arch sm_80`:

| arm | R | median kernel ms | amortized ns per warp reduction |
|---|---|---|---|
| `api_chain` | 1 | 1.346 | 8.41 |
| `shuffle_chain` | 1 | 4.350 | 27.19 |
| `api_independent` | 4 | 1.853 | 2.90 |
| `shuffle_independent` | 4 | 5.769 | 9.01 |

Paired on one target and one binary: **3.23×** for the dependency chain and
**3.11×** for the independent mix. Scaling, `api_chain` / `shuffle_chain`:
0.048/0.145, 0.221/0.704, 0.866/2.800, 1.727/5.595 ms at 1k, 5k, 20k, 40k
iterations.

**Same dependency tree, `--arch sm_75`:** the control. `api_chain` and
`shuffle_chain` both report 27.19 ns, and the independent arms both 9.01 ns —
the API compiles to the butterfly and the two arms are the same code.

`predicate_kind=` is not reported: the per-kernel PTX path proof from the
previous revision of this PR is kept outside the diff and is still available.

## Reproducing

```bash
cargo oxide run warp_reduce_redux_bench --arch sm_80   # on the tree under test
cargo oxide run warp_reduce_redux_bench --arch sm_75   # in a separate build tree
cargo oxide fmt --check
bash scripts/sync-example-locks.sh --check
bash scripts/check-example-smoketest-contract.sh
```

Needs #1305 for the `sm_80` arm to be a `redux.sync` build; without it both arms
are the butterfly and the harness still runs.

## Limitations

- One A100 SKU and one driver. Nothing here extrapolates to another device.
- `timer_kind` is a host wall clock, not CUDA events: this host API path was
  verified against the pinned `cuda-core` and the event type is available, but
  the harness keeps one timing path rather than two. The figure is bounded by
  that, which is why it is named an amortised cost.
- No executed SASS: `cargo oxide inspect` stops at PTX and the driver JITs it at
  load, so path evidence is PTX.
- `sm_75` is compile and runtime coverage on an A100 host, not a Turing GPU run.
- The two known scanner gaps in #1304 were not re-tested here.

## Scope

Measurement harness and documentation. It does not touch `crates/cuda-device`,
does not change the floor rule, and does not restate #1305's implementation,
which stays with its author and its #1304 dependency. Raw experiment artifacts
are kept outside this diff.
