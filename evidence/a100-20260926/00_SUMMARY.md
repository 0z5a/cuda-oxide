# cuda-oxide · A100 / SM80 execution — results

Pack: `cuda_oxide_A100_execution_2026-09-26` (A1 #399, A2 #811/#1305, A3 #1235, A4 #1328).
Target repo: `NVlabs/cuda-oxide`. Frozen base: `ec4aa4797956534578a1af010f86252a0b6d8626` (main).

## Deliverables

| item | outcome | artefact |
|---|---|---|
| **A4** #1328 | fix + 5 regressions + A100 device oracle, all green | draft PR [#1338](https://github.com/NVlabs/cuda-oxide/pull/1338) · [`evidence/A4/conclusion.md`](A4/conclusion.md) |
| **A2** #811/#1305 | A100 correctness + sanitizers + paired measurement | draft PR [#1339](https://github.com/NVlabs/cuda-oxide/pull/1339) · [`evidence/A2/conclusion.md`](A2/conclusion.md) |
| **A1** #399 | audit + test-only regression (no implementation) | draft PR [#1340](https://github.com/NVlabs/cuda-oxide/pull/1340) · [`evidence/A1/conclusion.md`](A1/conclusion.md) |
| **A3** #1235 | audit only; no PR is warranted by the gates | [`evidence/A3/conclusion.md`](A3/conclusion.md) |

All three PRs are drafts against `main`, from branches on the `0z5a` fork:
`0z5a`, `0z5a-a2-redux-a100-validation`, `0z5a-a1-iterator-local-array`.
Nothing was pushed to `main` and no comment, claim or assignment was posted to
any issue or PR.

## Headline numbers

### A2 — `redux.sync` against the shuffle butterfly (A100-SXM4-40GB, cc 8.0)

Same source, two builds; the form is chosen by the compile-time floor cfg, so
the comparison is across builds. At `sm_80` the pair is also compared inside one
binary, against a hand-written butterfly, in one interleaved pass.

| workload | `redux.sync` (ns/reduction) | butterfly (ns/reduction) | speedup `sm_80` |
|---|---|---|---|
| dependent chain (latency) | 8.4 – 10.8 | 27.2 – 35.0 | **3.24×** |
| four independent reductions (throughput) | 2.9 – 3.7 | 9.4 – 12.1 | **3.24 – 3.27×** |

Three independent batches agree on the ratio to the printed digit; the `sm_75`
control build times the two arms identically (27.2 = 27.2, 9.4 = 9.4), which is
the evidence the harness measures the reduction and nothing else. Cross-build
(`--arch sm_80` vs `--arch sm_75`): 2.5×–3.2×.

Path proof: per-kernel PTX shows `redux.sync.add.s32` in the API kernel with no
`shfl`, and five `shfl.sync.bfly.b32` + `add.s32` in the control with no `redux`.

This is a primitive-level number. The reduction goes from ten instructions to
one, so the call-site gain is bounded by Amdahl and is not an end-to-end or
serving figure.

### A4 — nothing to speed up

Correctness item; `NOT_APPLICABLE` for performance. Reported as evidence only.

## What the A100 actually ran

| run | result |
|---|---|
| A4 `redux_shift_regression` `--arch sm_80` | all 128 lanes match the host; memcheck 0 errors; synccheck 0 errors |
| A2 `warp_reduce_redux` `--arch sm_80` (PR oracle) | 4/4 cases correct; memcheck 0 errors; synccheck 0 errors |
| A2 `warp_reduce_redux` `--arch sm_75` | compiles; PTX keeps the shuffle fallback |
| A2 paired benchmark, 3 batches × 2 builds | see table |
| A1 `iterator_local_array_regression` `--arch sm_80` | both kernels match the host; memcheck 0 errors; code-shape check PASS |
| `cargo test -p mir-lower --all-targets` at A4 head | 304 + 152 pass, 0 fail |
| Falsification (A4, pre-#1264 lowering restored) | 2 tests fail with the report's `LargerThanOperand` error |

## Environment

| item | value |
|---|---|
| GPU | 2× `NVIDIA A100-SXM4-40GB`, cc 8.0, driver `595.91.07`, no MIG (runs used device 0) |
| host | Ubuntu 24.04.4, 192 vCPU, 251 GiB RAM, 1.1 TB overlay |
| rustc | `nightly-2026-08-28-x86_64-unknown-linux-gnu` (repo pin), with `rust-src`, `rustc-dev`, `llvm-tools`, `clippy`, `rustfmt` |
| LLVM | `llc` `23.1.0-rust-1.100.0-nightly` from that toolchain |
| CUDA | toolkit `13.0.88`, `libNVVM 2.0`, `nvJitLink 13.0`, `libdevice.10.bc`, compute-sanitizer `2025.3.1.0` |
| clang | `clang-21` / `libclang-21` (bindgen for the host runtime) |
| isolation | `RUSTUP_HOME`/`CARGO_HOME` under `/workspace/0z5a`; the box's CUDA 12.8 install was left untouched and CUDA 13 was added alongside it |

## Honesty notes

- **A4 does not claim to fix a live failure.** The reported reproducer already
  passes on main because #1264 normalises reduction results; this closes the
  latent hole in `convert_shift` and adds the regression that was missing. The
  PR body says so.
- **A1 does not reopen #399.** Its shape is handled on main; only the regression
  was missing. `CLOSURE_COMMIT` is recorded as `UNKNOWN` rather than guessed.
- **A3 got no PR** because its own gates (G1 address-path A/B, G2 source
  schedule, G3 compiler carrier) are all unmet — `NOT_RUN` / `NOT_ESTABLISHED` /
  `NOT_AUTHORIZED_BY_EVIDENCE`. Opening a transform PR there would be
  implementing an unproven change.
- **Executed SASS was not captured.** `cargo oxide inspect` stops at PTX and the
  driver JITs it at load, so every code-path claim here is about PTX, not about
  the cubin that ran.
- `sm_75` results are compile/PTX coverage on an A100 host, never a Turing GPU
  run. No B200, A10G or RTX 5090 number is presented as an A100 number.

## Layout

```
evidence/
  00_SUMMARY.md          this file
  A1/{closure-audit.md, conclusion.md}
  A2/{conclusion.md, a2.log, a2bench2-full.log, bench-ptx-sm80.txt, a2-ptx-sm80.txt, a2-ptx-sm75.txt}
  A3/{coverage-audit.md, conclusion.md}
  A4/{conclusion.md, a100-run-raw.log, a4-inspect-sm80.txt, a4_falsify.log, a4verify.log}
  ../pr/{a1-body.md, a2-body.md, a4-body.md}
  ../upstream/{issue-*.json, pr-*.json}
  ../task/                 the execution pack, verbatim
  ../oxide-mac/            checkout used for the source work
```
