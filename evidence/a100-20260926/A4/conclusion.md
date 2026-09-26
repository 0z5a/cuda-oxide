# A4 · #1328 shift lowering — result

**Status: fix + regression + A100 device oracle, all green.**
Target: `NVlabs/cuda-oxide`. Base: `ec4aa4797956534578a1af010f86252a0b6d8626` (main).
Hardware: 1× `NVIDIA A100-SXM4-40GB`, cc 8.0, driver `595.91.07`, no MIG.

## What the report actually was

`convert_shift` (`crates/mir-lower/src/convert/ops/arithmetic.rs`) contains the
containment logic for a shift whose two operands differ. It compared *widths*:

```rust
let cast_op = if lhs_width > rhs_width { zext } else { trunc };
```

The `else` arm assumes `lhs_width > rhs_width` was false because the RHS is
wider. It is also reached when the widths are **equal** and only the dialect
*representation* differs (`ui32` against a signless literal), and then it emits
`llvm.trunc i32 -> ui32` — which LLVM rejects, because `trunc` needs a strictly
smaller result. The whole device module failed verification.

## P0 — is it reproducible on main?

**No. The reported reproducer is already fixed, by a different change.**

| Evidence | Result |
|---|---|
| `git merge-base --is-ancestor fc3ec002 HEAD` | YES — the report SHA is an ancestor of main |
| `git log -S "would otherwise declare the intrinsic as returning"` | `8bc013b7` `fix(mir-lower): lower redux.sync results to signless integers (#1264)`, 2026-09-18 |
| `git show fc3ec002:crates/mir-lower/src/convert/intrinsics/warp.rs` | `convert_redux` used the dialect result type raw: `let result_ty = op.deref(ctx).get_result(0).get_type(ctx);` |
| same file at main | `let result_ty = convert_type(ctx, mir_result_ty)...` |

At `fc3ec002` the `redux.sync` declaration was emitted as
`llvm.func <builtin.integer ui32(i32, i32)>`, so the reduction result *was* a
`ui32` value and the shift hit the equal-width arm. `convert_type` on the LLVM
side maps every `IntegerType` to `Signedness::Signless`
(`crates/mir-lower/src/convert/type_interface_impls.rs:176-186`), so after #1264
no reduction result carries a representation into the shift.

**Falsification.** Restoring exactly that one line in a detached worktree at
main makes the new tests fail with the reported error:

```
crates/mir-lower/tests/lowering_test/shift_representation.rs:179:
lowered module must verify: Error { kind: VerificationFailed,
                                   err: LargerThanOperand, loc: Unknown }
test result: FAILED. 3 passed; 2 failed
```

`LargerThanOperand` is `TruncOp`'s verifier (`integer_cast_verify(... ULT)`,
pliron-llvm `ops.rs:3305`), i.e. the same "Result type must be smaller than
operand type" the issue reports. With main's lowering the same 5 tests pass.

## Reachability of the remaining defect

The equal-width arm is **not reachable through the current pipeline**. Every
integer that reaches an LLVM-dialect op is produced by `convert_type`, which is
total on integers and always signless; I checked every intrinsic converter that
derives a result type from the dialect op:

| Site | Verdict |
|---|---|
| `convert/intrinsics/warp.rs:292` (`convert_redux`) | normalised via `convert_type` |
| `convert/intrinsics/atomic.rs:548` | normalised via `convert_type` |
| `convert/intrinsics/memory.rs:41` | rebuilt as `Signedness::Signless` |
| `convert/intrinsics/asm.rs:78,103` | normalised via `convert_type` |
| `convert/ops/constants.rs` | normalised, pinned by `convert_integer_preserves_bits_and_makes_type_signless` |
| `generated_intrinsics/mod.rs:69` | constructs signless directly |

So the branch is dead **and wrong**. It stays in the tree as a trap: any future
path that produces a represented integer (which is exactly what #1264 removed)
re-emits an invalid module rather than a legal representation change. The
report's own framing — "the case where the widths are equal but the types are
not is missed" — is what this fixes.

## The fix

`crates/mir-lower/src/convert/ops/arithmetic.rs`, two changes that are only
correct together:

1. `convert_shift` — a three-way match on width. Equal width with unlike types
   takes `llvm::BitcastOp`, the same same-width representation change the cast
   lowering already uses (`convert/ops/cast.rs:228`, `IntToInt (same width)
   -> bitcast`). It carries the same bits; it does not narrow or widen.
2. `mask_shift_amount` — the `bit_width - 1` mask constant was built with a
   fresh `Signedness::Signless` type. It now carries the shift value's own
   representation. Without this the count-`and` and the shift itself would have
   mismatched operand types: `llvm.and` and `llvm.shl` are
   `SameOperandsAndResultType` ops (`pliron-llvm/src/ops.rs:206-225`) and
   `SameOperandsType::verify` rejects unequal operands
   (`pliron/src/builtin/op_interfaces.rs:952-969`). This is why the "just
   `return rhs` on equal width" shortcut the task document warns about does not
   work — it moves the failure from the cast to the `and` and the shift.

The widened result is legal LLVM: `llvm-as` accepts `bitcast i32 %x to i32`
(verified with the pinned toolchain's `llvm-as`, LLVM 23.1.0-rust), and the
exporter prints every `IntegerType` as `i{width}`
(`crates/llvm-export/src/export/types.rs:64-65`), so the two dialect types are
one LLVM type.

Files:
- `crates/mir-lower/src/convert/ops/arithmetic.rs` (+21/−8)
- `crates/mir-lower/tests/lowering_test/shift_representation.rs` (new, 422 lines, 5 tests)
- `crates/mir-lower/tests/lowering_test/main.rs` (+1 module)
- `crates/rustc-codegen-cuda/examples/redux_shift_regression/` (new example + `Cargo.toml`)

## Tests

`cargo test -p mir-lower --test lowering_test shift` → **5 passed; 0 failed.**

| test | what it pins |
|---|---|
| `redux_max_u32_shift_lowers` | the #1328 shape end to end: reduction result `<<` literal, no `trunc`, no `zext`, `llvm.shl` operands and result share one type |
| `shift_count_mask_shares_the_shift_operand_type` | the mask `and` shares that type and the shift consumes its result |
| `shift_count_widths_convert_in_the_documented_direction` | 32←8 widens by `zext`, 8←32 narrows by `trunc`, neither is a representation change |
| `right_shift_selects_arithmetic_or_logical_from_mir_signedness` | signed MIR value → `ashr`, unsigned → `lshr`, both keep the mask |
| `matching_count_width_is_masked_without_a_cast` | equal width: no cast at all, mask is `bit_width - 1`, shift consumes it |

`redux_max_u32_shift_lowers` and `shift_count_mask_shares_the_shift_operand_type`
are the two that fail under the pre-#1264 lowering.

## A100 device evidence

`crates/rustc-codegen-cuda/examples/redux_shift_regression` — 4 kernels, 3 input
patterns (`ramp`, `extremes`, `random` from a fixed-seed LCG), 128 threads
(4 warps) so cross-warp leakage is covered, every lane writes its own result.

```
cargo oxide build    redux_shift_regression --arch sm_80   # ok
cargo oxide run      redux_shift_regression --arch sm_80   # SUCCESS
cargo oxide sanitize redux_shift_regression --arch sm_80 --tool memcheck   # 0 errors
cargo oxide sanitize redux_shift_regression --arch sm_80 --tool synccheck  # 0 errors
```

| case | result |
|---|---|
| `ramp` / `extremes` / `random`: `redux.sync.max.u32 << c` | all 128 lanes match host |
| same inputs: `>> c` stays logical | all 128 lanes match host (bit 31 set on `extremes`) |
| `redux.sync.min.s32 >> c` stays arithmetic | all 128 lanes match host (answers negative) |
| `wrapping_shl` with counts 32/33/63 | mask to 0/1/31, all 128 lanes match host |

### Emitted PTX (`cargo oxide inspect redux_shift_regression --arch sm_80`)

```text
 76:  redux.sync.max.u32  %r1, %r9, %r10;     # shl kernel
 83:  and.b32            %r12, %r11, 31;      # count mask
 84:  shl.b32            %r13, %r1, %r12;
210:  redux.sync.min.s32  %r1, %r9, %r10;     # signed control
218:  shr.s32            %r13, %r1, %r12;     # arithmetic
277:  redux.sync.max.u32  %r1, %r9, r10;
285:  shr.u32            %r13, %r1, %r12;     # logical
```

The signedness selection is visible in the PTX (`shr.s32` for the signed
control, `shr.u32` for unsigned), the count mask is present, and no `.local`,
`ld.local` or `st.local` appears in the module.

Evidence level: real A100 execution of a real cubin, memcheck and synccheck
clean. It is not a correctness proof for inputs outside the four enumerated
patterns, and it is not a performance claim — #1328 is a correctness item and
`NOT_APPLICABLE` for speedup.

## Evidence index

| file | content |
|---|---|
| `a100-run-raw.log` | full build + run + memcheck + synccheck transcript |
| `a4-inspect-sm80.txt` | emitted PTX for all four kernels |
| `falsify.log` | pre-#1264 lowering: 2 tests fail with `LargerThanOperand` |
| `verify.log` | `cargo test -p mir-lower --all-targets`, fmt, clippy |
| `manifest.json` | SHAs, toolchain, GPU identity |

## Limitations

- The equal-width representation branch has no test that *executes* it: the
  pipeline cannot produce a represented integer on main, so the branch is
  covered by inspection plus the `SameOperandsAndResultType` reasoning above,
  not by a passing test. Recorded as `UNVERIFIED_BY_TEST` for that arm only.
- The reported reproducer was verified fixed at the *lowering* level and by the
  device example. It was not rebuilt against `fc3ec002`, which has its own
  toolchain pin; the falsification above reproduces its behaviour instead.
- Closure of the upstream issue is not claimed. #1328 is a report by a third
  party who offered a patch; the PR body says `Related to #1328`, not `Fixes`.
