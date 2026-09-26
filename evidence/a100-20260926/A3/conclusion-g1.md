# A3 · #1235 — G1 result: the two shared-address forms are equivalent

**Status: G1 measured and green. The answer is a negative one, and that is the
useful part.**

Gate G1 asked: for one shape, one layout and one row, do the pointer form and
the `.shared::cta.u32` form of `ldmatrix` hand back the same fragment on A100?

**Yes — bit for bit, in all twelve combinations, at three row offsets.**

## How it was measured

New example `crates/rustc-codegen-cuda/examples/ldmatrix_address_form_ab`. Each
kernel loads the same shared rows twice, in one instruction stream:

- once through `ldmatrix_x{1,2,4}[_trans](ptr)`, with element arithmetic on a
  per-lane pointer;
- once through `ldmatrix_x{1,2,4}[_trans]_shared_u32(addr)`, where `addr` is
  `cvta_generic_to_shared_u32` applied **once** to the shared base and then
  advanced with byte arithmetic in u32 — exactly the shape #1235 proposes.

The two forms differ only in how the row address is expressed. Same shape, same
layout, same lane, same row.

This matters because the existing oracle (`generated_ldmatrix`) executes two of
the twelve combinations and checks each **only against a host model, never
against the other**. A green `generated_ldmatrix` therefore could not answer G1.

Both forms are additionally compared against a host model, so a shared fault in
the two forms cannot pass as agreement. Transposed shapes use the transpose of
the same element map. `row_offset` is a runtime kernel argument, so the address
arithmetic cannot fold to a constant.

## Result — A100-SXM4-40GB, `--arch sm_80`

18 cells: twelve shape combinations × offsets {0, 8, 32 rows}.

| shape | offset | pointer vs native | both vs host model |
|---|---|---|---|
| x1 | 0 / 8 / 32 | equal | match |
| x1.trans | 0 / 8 / 32 | equal | match |
| x2 | 0 / 8 / 32 | equal | match |
| x2.trans | 0 / 8 / 32 | equal | match |
| x4 | 0 / 8 / 32 | equal | match |
| x4.trans | 0 / 8 / 32 | equal | match |

```
SUCCESS: both address forms return identical fragments for every shape
```

`sm_75` is not applicable here: the example requires `sm_75+` and this run is at
`sm_80`. `compute-sanitizer` was not run for this example.

## What this says about the rest of #1235

The task document's own gate table says that when G1 shows no difference, the
result should be recorded as **equivalence / no gain** and not packaged as a
performance improvement. That is what this is.

It also bears on **G3**: #1235's compiler layer proposed a canonical
counted-MMA carrier to keep a proven counted register-MMA loop in one lowering
lifetime, on the grounds that pointer indexing leaves wide address recurrences
and conversions in the loop. G1 says the two address forms produce identical
fragments, so **the address form alone is not a measurable gap on this
hardware**. That does not measure the instruction-count difference inside a
counted loop — G1 compares fragments at a single load, not loop-level codegen —
so it is evidence against the address-width motivation, not a refutation of
every possible carrier.

**G2 (source-level scheduling) and G3 (compiler residual) were not run.** They
were not attempted, and no conclusion should be read into them from this
report.

## Coverage closed on the way

The audit behind this (`A3/coverage-audit.md` in the earlier revision) found
that the twelve combinations existed in source but only two were ever executed
and checked, the other ten living inside `legacy_ldmatrix_compile_oracle`, which
`main()` never launches. It also found that wiring that kernel up as-is would
write `32 × LEGACY_REGISTERS = 896` words into a buffer sized at 128 —
`get_unchecked_mut` only `debug_assert!`s, so a release build would write out of
bounds. This example does not reuse that kernel.

## Limitations

- One A100 SKU, one driver, one target (`sm_80`).
- Fragment equality at the load, not instruction-count or register-pressure
  equality inside a loop. The PTX for the two address forms was not diffed for
  this report.
- No `compute-sanitizer` run for this example (memcheck and synccheck were run
  for the other three items).
- x1/x2 consume only the first eight or sixteen lanes' addresses; the remaining
  lanes hold valid aligned addresses by construction but are not consumed.
- The host model is derived here, not shared with `generated_ldmatrix`; the two
  agree on the x4 normal case that both cover.
