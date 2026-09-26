# A100 / SM80 execution record — 2026-09-26

Work for the four items in the cuda-oxide A100 execution pack, plus the G1
follow-up, on one branch. Nothing here targets `main`.

| item | upstream | outcome | where |
|---|---|---|---|
| A4 | issue #1328 | shift representation fix + direct converter regressions + device oracle — draft PR #1338 (`0z5a`) | `A4/` |
| A2 | issue #811 / PR #1305 | paired warp-reduction measurement harness — PR #1339, closed by the author | `A2/` |
| A1 | issue #399 | iterator local-array regression + PTX shape checker — draft PR #1340 (`0z5a-a1-iterator-local-array`) | `A1/` |
| A3 | issue #1235 | coverage audit, then gate G1 measured: the two shared-address forms are equivalent | `A3/` |

`00_SUMMARY.md` has the numbers. `pr/` holds the PR bodies as submitted.

Branch layout on the `0z5a` fork:

| branch | contents |
|---|---|
| `0z5a` | A4 — PR #1338 |
| `0z5a-a1-iterator-local-array` | A1 — PR #1340 |
| `0z5a-a2-redux-a100-validation` | A2 harness — PR #1339, closed |
| `0z5a-work` | this branch: all four examples plus the reports |

Raw evidence directories were moved out of the PR diffs before review and kept
here, with a local archive and SHA-256 per PR head.

Every number in these reports comes from a real A100-SXM4-40GB (cc 8.0,
driver 595.91.07). Nothing is projected, and the places where a claim is *not*
made — executed SASS, end-to-end gain, `sm_75` as a Turing run, the closure
commit for #399, gates G2/G3 for #1235 — are called out in the individual
conclusions.
