## What

`convert_shift` chose how to reconcile the shift value and the shift count by
comparing **widths**:

```rust
let cast_op = if lhs_width > rhs_width { zext } else { trunc };
```

The `trunc` arm is documented as the "count is wider" case, but it is also
reached when the widths are *equal* and only the dialect representation differs.
`llvm.trunc i32 -> ui32` then fails verification — LLVM requires a strictly
smaller result — and the whole device module fails with it. That is #1328.

Two changes, and they are only correct together:

- equal width with unlike representations now takes `llvm.bitcast`, the
  same-width representation change the cast lowering already uses for
  `IntToInt (same width)` (`convert/ops/cast.rs`). Returning the count unchanged
  is not enough: `llvm.and` and `llvm.shl` are `SameOperandsAndResultType` ops,
  so the mask and the shift would then disagree with the value.
- `mask_shift_amount` builds the `bit_width - 1` mask with the shift value's own
  representation instead of a fresh signless type.

The bitcast is a no-op in the emitted LLVM IR: the exporter prints every
`IntegerType` as `i{width}` (`llvm-export/src/export/types.rs`), and the pinned
`llvm-as` accepts `bitcast i32 %x to i32`.

## Status of the reported reproducer

**The reproducer no longer fails on main, and this does not change that.**
`convert_redux` used to declare the reduction as returning the catalog's `ui32`,
so the reduction result was the represented value that reached the shift.
`#1264` (`8bc013b7`, *lower redux.sync results to signless integers*) started
translating that result through `convert_type`, and every other intrinsic
lowering path already normalises too, so the equal-width arm is not reachable
from source today.

It is still wrong, and nothing enforces the invariant it depends on. This
change makes the containment total, and adds the regression that was missing:
no test ran an intrinsic result into a shift, so the shape the report is about
had no coverage even after it started passing.

Reverting the `#1264` translation in a scratch worktree reproduces the report
against these tests:

```
crates/mir-lower/tests/lowering_test/shift_representation.rs:179:
lowered module must verify: Error { kind: VerificationFailed,
                                   err: LargerThanOperand, loc: Unknown }
test result: FAILED. 3 passed; 2 failed
```

`LargerThanOperand` is `TruncOp`'s verifier, i.e. the same *"Result type must be
smaller than operand type"* from the issue.

Related to #1328.

## Tests

`cargo test -p mir-lower --all-targets`: 304 + 152 pass, 0 fail.
`cargo oxide fmt --check` clean on the touched files.

New in `crates/mir-lower/tests/lowering_test/shift_representation.rs`:

| test | pins |
|---|---|
| `redux_max_u32_shift_lowers` | the reported shape end to end: no `trunc`, no `zext`, `llvm.shl` operands and result share one type |
| `shift_count_mask_shares_the_shift_operand_type` | the mask `and` shares that type and the shift consumes its result |
| `shift_count_widths_convert_in_the_documented_direction` | 32←8 widens by `zext`, 8←32 narrows by `trunc`, neither is a representation change |
| `right_shift_selects_arithmetic_or_logical_from_mir_signedness` | signed MIR value → `ashr`, unsigned → `lshr`, both keep the mask |
| `matching_count_width_is_masked_without_a_cast` | equal width: no cast, mask is `bit_width - 1`, shift consumes it |

## Device evidence

New example `redux_shift_regression` (4 kernels, 128 threads = 4 warps, three
input patterns, every lane writes its own result, host oracle).

| item | result |
|---|---|
| `cargo oxide build --arch sm_80` | ok |
| `cargo oxide run --arch sm_80` | SUCCESS, all lanes match host |
| `sanitize --tool memcheck` | 0 errors |
| `sanitize --tool synccheck` | 0 errors |
| HW | A100-SXM4-40GB, cc 8.0, driver 595.91.07, CUDA 13.0.88 |

`>>` on the unsigned reduction stays logical on a warped input carrying
`0xffffffff`; `>>` on `redux.sync.min.s32` stays arithmetic on negative lanes;
`wrapping_shl` with counts 32/33/63 masks to 0/1/31. Emitted PTX:

```text
redux.sync.max.u32  %r1, %r9, %r10;
and.b32             %r12, %r11, 31;
shl.b32             %r13, %r1, %r12;
redux.sync.min.s32  %r1, %r9, %r10;
shr.s32             %r13, %r1, %r12;
shr.u32             %r13, %r1, %r12;
```

## Performance

NOT_APPLICABLE. This is a correctness fix; no benchmark is claimed and none was
made for it.

## Limitations

- The equal-width representation arm has no test that executes it, because the
  pipeline cannot produce a represented integer value on current main. It is
  covered by inspection and by the `SameOperandsAndResultType` argument above,
  not by a passing test.
- One A100 SKU, one driver. `sm_75` and other backends were not exercised for
  this change; the shift lowering is target-independent.
- The `libNVVM` backend route was not run.

## Out of scope

No change to intrinsic lowering, no default-policy change, no toolchain bump.
The `#1264` translation of reduction results is left as it is.
