# A3 · #1235 — result

**Status: audit complete; no implementation. The three gates the task defines are
answered `NOT_RUN` / `NOT_ESTABLISHED` / `NOT_AUTHORIZED_BY_EVIDENCE`, which is
the outcome the task document allows.**

Full audit: [`coverage-audit.md`](coverage-audit.md) (298 lines, read-only,
frozen at `ec4aa4797956534578a1af010f86252a0b6d8626`).

## What is already done upstream

The slice this task must not repeat is merged: `a07f804a` *feat(wmma): add
shared-u32 ldmatrix forms (#1236)* is an ancestor of main. All fifteen API
items exist — six `ldmatrix_x{1,2,4}[_trans]_shared_u32`, their six pointer
twins, `mma_m16n8k16_f32_bf16`, and the two `cvta_generic_to_shared_*`
conversions.

The structural fact that decides the rest: **the `*_shared_u32` names are not
separate catalog entries.** They are `compatibility_paths` on the same six
entries as their pointer twins (`intrinsics/catalog.json`, ids `i0025`–`i0029`,
`i0013`; overlay `intrinsics/overlay/ldmatrix.toml`). Address form is resolved
only inside the lowering boundary (`crates/mir-lower/src/convert/intrinsics/ldmatrix.rs:109-151`),
so anything that compares the two forms has to compare *lowered code*, not
catalog rows.

## Gate results

| gate | question | answer | why |
|---|---|---|---|
| G1 · address path | do pointer and shared-u32 differ on A100 for the same schedule? | **NOT_RUN** | the existing oracle never compares the two forms against each other, only against the same host model; an A/B needs a new same-shape kernel pair |
| G2 · source schedule | is source-level fragment scheduling enough? | **NOT_ESTABLISHED** | depends on G1 |
| G3 · compiler carrier | is a counted-MMA carrier warranted? | **NOT_AUTHORIZED_BY_EVIDENCE** | depends on G1 and G2 |

## Existing oracle coverage: 2 of 12

`crates/rustc-codegen-cuda/examples/generated_ldmatrix` executes and
host-checks exactly two of the twelve shape × transpose × address-form
combinations:

| covered | evidence |
|---|---|
| x4 / normal / shared-u32 | `main.rs:112-113`, launched `:305-314`, checked `:321-336` |
| x4 / normal / pointer | `main.rs:156`, `:197`, launched `:344-355`, `:383-395` |

The other ten are present in source inside `legacy_ldmatrix_compile_oracle`
(`main.rs:211-276`), which `main()` **never launches and never checks** — the
symbol appears only at its own definition. So "all twelve compile" is source
presence, not coverage. `sm_75` compile is `UNVERIFIED` for every row: the gate
admits `sm_80` (`generated_intrinsic_targets/ldmatrix.rs:33-58`) but every
recorded artefact is `sm_75` and CI drives the example at `sm_90` through
`emit-ltoir`.

**Hazard found while auditing that dead kernel:** it writes
`32 × LEGACY_REGISTERS = 896` words (`main.rs:42`, `:245-275`) into a buffer
`main()` sizes at 128 words (`:39`, `:297-298`). `get_unchecked_mut` only
`debug_assert!`s (`cuda-device/src/disjoint.rs:434-441`), so wiring it up as-is
in a release build writes out of bounds. Any future work on this oracle has to
size that buffer first.

`libNVVM` caveat for the same shape: the x4 entry records a **failed**
`typed_nvvm` stage at `compute_75`–`compute_90a`
(`NVVM_ERROR_COMPILATION(9): unsupported operation`), succeeding only at
`compute_100`/`120`. The shipped route for x4 is `inline_ptx`; x1/x2 have no
such failure.

## Why no compiler transform

The task's stop conditions are met. There is no demonstrated residual codegen
defect: the address-path question has not been measured, and #1237's discussion
already showed the originally reported problem is addressable in ordinary Rust
source. Writing a counted-MMA carrier now would be implementing an unproven
transform, which the gate list forbids.

## Cheapest next step

A single device oracle that loads the **same** shape and layout twice — once
through the pointer form and once through the shared-u32 form — into the same
fragment mapping, on the same hardware, and compares the fragments against each
other. That is the missing cell for every one of the twelve rows at once, and it
is host-model-free evidence for G1.

## Limitations

- No build, no GPU run in this audit: it is source and catalog evidence only.
  Every device cell is `NOT_RUN`.
- The `sm_80` compile cells are `UNVERIFIED`, not unsupported: the target gate
  admits `sm_80`, but no artefact in the tree is at that target.
- The catalogue/overlay facts are pinned to `ec4aa479`. The upstream repo moves.
