## What this changes

`convert_shift` reconciled the shift value and the shift count by comparing
**widths**:

```rust
let cast_op = if lhs_width > rhs_width { zext } else { trunc };
```

The `trunc` arm is documented as "the count is wider", but it is also reached
when the widths are *equal* and only the dialect representation differs.
`llvm.trunc i32 -> ui32` then fails verification — LLVM requires a strictly
smaller result — and the whole device module fails with it. That is #1328.

The revision handles representation explicitly, on two rules that are both
required:

- **The shift value is unified to the signless representation first.** Every op
  in the chain is an `IntBinArithOp`, and pliron-llvm rejects a represented
  (`uiN` / `siN`) operand of one outright — *"Integer binary arithmetic Op can
  only have signless integer result/operand type"*
  (`pliron-llvm/src/op_interfaces.rs:94-114`). Carrying a represented value into
  the count mask cannot produce a legal module, whatever the count looks like.
  The change is a same-width change of the same bits.
- **A count that differs only in representation takes that same change**, not a
  truncation. Handing the count over unchanged is not enough either: the mask and
  the shift are `SameOperandsAndResultType` ops, so the whole chain has to agree
  on one representation.

The bitcast is a no-op in the emitted LLVM IR: the exporter prints every
`IntegerType` as `i{width}` (`llvm-export/src/export/types.rs:64-65`), and the
pinned `llvm-as` accepts `bitcast i32 %x to i32`.

## Status of the reported reproducer

**The reproducer already passes on main, and this revision does not change
that.** `convert_redux` used to declare the reduction as returning the catalog's
`ui32`, so the reduction result was the represented value that reached the
shift. #1264 (`8bc013b7`) started translating it through `convert_type`, and
every other intrinsic lowering path already normalizes, so no source construct
reaches the illegal arm today.

The arm stayed illegal. This revision fixes it and adds the coverage that
reaches it — the previous revision only had integration tests that pass either
way.

Related to #1328.

## Direct converter coverage

`convert_shift` cannot be reached with represented operands from Rust source,
because the pipeline normalizes both operands first. The count normalization is
now a private helper the production path calls, and the tests drive it, the real
count mask, and the real `convert_shl` / `convert_shr` from operands built at
that boundary.

Each case asserts the whole chain rather than an opcode: the cast's operand and
result types, the mask constant's type and value, the `and`'s operands and
result, that the shift consumes the masked count, and that **every op the
conversion produced passes its own verifier**.

| case | value | count | asserted |
|---|---|---|---|
| `represented_value_is_unified_to_signless` | unsigned-32 | signless-32 | one representation change, no trunc/zext |
| `represented_count_is_unified_to_signless` | signless-32 | unsigned-32 | the mirror direction |
| `signed_value_is_unified_without_a_width_change` | signed-32 | signless-32 | no widening or narrowing |
| `identical_representations_introduce_no_cast` | signless-32 | signless-32 | no cast, mask still present |
| `narrower_count_is_zero_extended` | signless-32 | signless-8 | one `zext`, result on 32 bits |
| `wider_count_is_truncated` | signless-8 | signless-32 | one `trunc`, mask is 7 |
| `right_shift_signedness_comes_from_the_original_value` | signed / unsigned | signless-32 | `ashr` vs `lshr`, count still masked |

## Destructive controls

Each mutation was applied in a scratch copy, run, and reverted (the file hash
was checked before and after).

| mutation | expected | observed |
|---|---|---|
| equal-width count truncates again (the reported bug) | direct test rejects the illegal same-width trunc | `represented_count_is_unified_to_signless` **FAILED** |
| the shift value is never unified | represented-value chain fails the `IntBinArithOp` verifier | 3 tests **FAILED** |
| mask carries the value's representation | — | **NOT DISTINGUISHABLE**: after the value is unified, `lhs_ty` is always signless, so the two spellings are the same program. Not claimed as covered. |

## Integration and device results

| command | result |
|---|---|
| `cargo test -p mir-lower --all-targets` | 311 passed, 0 failed (lib) + 152 passed, 0 failed (lowering_test) |
| `cargo oxide fmt --check` on the touched files | clean |
| `scripts/sync-example-locks.sh --check` | 239 example locks agree |
| `scripts/check-example-smoketest-contract.sh` | OK, 203 examples carry a success marker |
| `cargo oxide run redux_shift_regression --arch sm_80` | SUCCESS, all 128 lanes match the host |
| `cargo oxide sanitize … --tool memcheck` | 0 errors |
| `cargo oxide sanitize … --tool synccheck` | 0 errors |

Hardware: A100-SXM4-40GB, cc 8.0, driver 595.91.07, CUDA 13.0.88, rustc
`nightly-2026-08-28`. Emitted PTX keeps `redux.sync.max.u32` /
`redux.sync.min.s32`, the `and.b32 …, 31` count mask, and `shr.s32` for the
signed control against `shr.u32` for the unsigned one.

The example now gives each warp different input values and starts every output
buffer at the bitwise inverse of the host oracle, so a lane the kernel never
writes cannot read back as a correct answer.

## Performance

NOT_APPLICABLE. Correctness fix; no benchmark was run and none is claimed.

## Limitations

- One A100 SKU, one driver, `sm_80` only. The lowering is target-independent,
  but that was not exercised on another target for this change.
- The `libNVVM` backend route was not run.
- The device example is a smoke test. The count-masking invariant is asserted by
  the converter tests, not by the kernel: `count & 31` and `wrapping_shl` would
  agree with the oracle even if the backend dropped the mask.
- No executed SASS was captured; the code-path evidence is PTX.

## Out of scope

No intrinsic API change, no change to `convert_redux`'s signless normalization,
no SM-floor change, no dependency on #1305, no unification of integer casts
beyond the shift count. Raw logs and the historical falsification transcript are
kept outside this diff.
