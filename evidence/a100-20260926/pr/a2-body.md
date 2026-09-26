## What

#1305 chooses the reduction form from the target floor at compile time, so the
two forms cannot be compared inside one binary — it has to be two builds of one
source. That harness did not exist, and the #1305 body says so: *"no new
speedup measurement was made."* This adds it, and the control it needs to be
trustworthy.

- `api_chain` / `api_independent` drive `warp_reduce::<u32, _>`. One is a
  dependency chain, so it exposes reduction latency; the other runs four
  unrelated reductions per iteration so the issue slots have something to
  overlap.
- `shuffle_chain` / `shuffle_independent` spell the butterfly out with
  `WarpCollective::shfl_xor`. At `sm_80` that pair isolates the reduction from
  the rest of the build — same target, same compiler, same binary — which the
  cross-build comparison cannot do.
- 128 threads, 40,000 iterations, 15 samples per arm, 3 warm-up rounds, all four
  arms sampled round-robin inside one pass so clock drift lands on every arm.
  Values are per reduction, amortised over a multi-millisecond kernel.
- The first iterations are compared against a host simulation, so a loop the
  compiler folded away cannot pass as a timing.

**Needs #1305.** Without `cuda_oxide_sm_at_least` both builds take the butterfly
and the harness still runs, it just has nothing to compare. The numbers below
were taken at #1305's head, `cf4c55c3`.

## Measurement

A100-SXM4-40GB (cc 8.0, driver 595.91.07, no MIG), rustc `nightly-2026-08-28`,
`llc` LLVM `23.1.0-rust-1.100.0-nightly`, CUDA `13.0.88`.

Each cell is `median ms / ns per reduction`; three independent batches.

| build | batch | api_chain | shuffle_chain | api_independent | shuffle_independent |
|---|---|---|---|---|---|
| `sm_80` | 1 | 1.731 / 10.8 | 5.599 / 35.0 | 2.384 / 3.7 | 7.718 / 12.1 |
| `sm_80` | 2 | 1.346 / 8.4 | 4.349 / 27.2 | 1.853 / 2.9 | 5.995 / 9.4 |
| `sm_80` | 3 | 1.732 / 10.8 | 5.599 / 35.0 | 2.384 / 3.7 | 7.718 / 12.1 |
| `sm_75` | 1 | 4.350 / 27.2 | 4.350 / 27.2 | 5.995 / 9.4 | 5.996 / 9.4 |
| `sm_75` | 2 | 4.349 / 27.2 | 4.349 / 27.2 | 5.995 / 9.4 | 5.995 / 9.4 |
| `sm_75` | 3 | 4.350 / 27.2 | 4.350 / 27.2 | 5.996 / 9.4 | 5.996 / 9.4 |

**Speedup at `sm_80`**, paired inside one binary and one interleaved pass:

| workload | `redux.sync` | butterfly | ratio |
|---|---|---|---|
| dependent chain | 8.4 - 10.8 ns | 27.2 - 35.0 ns | **3.24x** (all three batches) |
| four independent reductions | 2.9 - 3.7 ns | 9.4 - 12.1 ns | **3.24 - 3.27x** |

Cross-build, which is the framing the PR uses (`--arch sm_80` against
`--arch sm_75`): 8.4-10.8 ns against 27.2 ns, i.e. **2.5x - 3.2x**.

The control is exact: at `sm_75` the API path and the hand-written butterfly
agree to the printed digit on all six cells, which is what a build where
`HAS_FULL_WARP_FORM` is false has to do.

### Path proof

The timing is only meaningful if the arms really are the two forms. Per-kernel
PTX at `sm_80`:

```text
.visible .entry shuffle_chain(...)
    shfl.sync.bfly.b32  %r10, %r39, 16, 31, -1;
    add.s32             %r11, %r10, %r39;
    ... five rounds, no redux in this kernel ...
.visible .entry api_chain(...)
    redux.sync.add.s32  %r10, %r20, %r9;
    redux.sync.add.s32  %r11, %r10, %r9;
    ... no shfl in this kernel ...
```

Correctness of the PR oracle itself was re-run on the same machine:

| case | result |
|---|---|
| `u32` full warp, six ops | `[496, 0, 31, 0, 0xffffffff, 0xffffffff]` as expected |
| `i32` full warp over `-16..=15` | `[-16, -16, 15]` — signedness preserved |
| `f32` control | `[496.0, 0.0, 31.0]` — fallback kept |
| `WarpTile<16>` control | `[120, 120]` — sub-warp tiles not promoted |
| `sanitize --tool memcheck` / `--tool synccheck` | 0 errors |

## Scope

This is a measurement harness and a validation record. It does not touch
`crates/cuda-device`, does not change the floor rule, and does not restate the
implementation — that stays with #1305 and its #1304 dependency.

## Limitations

- Primitive only. The reduction is one instruction against ten, so the ratio is
  bounded by Amdahl at the call site; this is not an end-to-end or serving
  number, and a kernel that reduces once after a memory-bound pass sees far
  less.
- One A100 SKU and one driver. The A10G/B200 coverage in #1305 is not repeated.
- Registers, occupancy and SASS were not captured: `cargo oxide inspect` stops
  at PTX and the driver JITs it at load, so the path evidence is PTX, not
  executed SASS.
- `sm_75` is compile/PTX coverage on an A100 host, not a Turing GPU run.
- The two known scanner gaps in #1304 (whitespace/comment-separated opcode
  modifiers, a quoted `.file` path) were not re-tested here.
