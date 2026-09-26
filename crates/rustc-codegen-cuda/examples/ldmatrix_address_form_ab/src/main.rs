/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Address-form equivalence for `ldmatrix` (#1235, gate G1).
//!
//! `generated_ldmatrix` executes two of the twelve shape x transpose x
//! address-form combinations, and each of those two is checked only against a
//! host model — never against the other. So it cannot answer the question this
//! example exists for: for one shape, one layout and one row, do the pointer
//! form and the `.shared::cta.u32` form hand back the same fragment?
//!
//! Each kernel below loads the same shared rows twice, in the same instruction
//! stream:
//!
//!   * once through `ldmatrix_x{1,2,4}[_trans](ptr)`, with element arithmetic
//!     on a per-lane pointer;
//!   * once through `ldmatrix_x{1,2,4}[_trans]_shared_u32(addr)`, where `addr`
//!     is `cvta_generic_to_shared_u32` applied **once** to the shared base and
//!     then advanced with byte arithmetic in u32 — the shape #1235 proposes.
//!
//! The two forms therefore differ only in how the row address is expressed.
//! Same shape, same layout, same lane, same row: any difference in the fragments
//! is a difference in address translation.
//!
//! Both are additionally compared against a host model, so a shared bug in the
//! two forms cannot pass as agreement. Transposed shapes use the transpose of
//! the same element map.
//!
//! `row_offset` is a runtime argument, so the address arithmetic cannot fold to
//! a constant, and the example runs every shape at offset 0 and at a non-zero
//! offset that still leaves a full tile inside the allocation.
//!
//! Build and run with:
//!   cargo oxide run ldmatrix_address_form_ab --arch sm_80

use cuda_core::simt::LaunchConfig;
use cuda_core::{CudaContext, DeviceBuffer};
use cuda_device::shared::cvta_generic_to_shared_u32;
use cuda_device::wmma::{
    ldmatrix_x1, ldmatrix_x1_shared_u32, ldmatrix_x1_trans, ldmatrix_x1_trans_shared_u32,
    ldmatrix_x2, ldmatrix_x2_shared_u32, ldmatrix_x2_trans, ldmatrix_x2_trans_shared_u32,
    ldmatrix_x4, ldmatrix_x4_shared_u32, ldmatrix_x4_trans, ldmatrix_x4_trans_shared_u32,
};
use cuda_device::{DisjointSlice, SharedArray, cuda_module, kernel, thread};
use cuda_intrinsics::sreg::thread_idx_x;

const LANES: usize = 32;
const MATRICES: usize = 4;
const ROWS: usize = 8;
const COLUMNS: usize = 8;
const WORDS_PER_ROW: usize = COLUMNS / 2;
const ROW_BYTES: usize = WORDS_PER_ROW * 4;

/// Byte offset of a row from the shared base, in the u32 address domain.
#[inline(always)]
const fn row_byte_offset(row_index: usize) -> u32 {
    (row_index * ROW_BYTES) as u32
}

/// Rows in the shared tile: one full x4 tile (32 rows) plus enough slack for the
/// largest non-zero offset the host uses.
const SHARED_ROWS: usize = 64;
const SHARED_WORDS: usize = SHARED_ROWS * WORDS_PER_ROW;

/// Largest offset that still leaves 32 whole rows above it.
const MAX_ROW_OFFSET: u32 = (SHARED_ROWS - MATRICES * ROWS) as u32;

/// A distinct nonzero value for every matrix element.
///
/// ```text
/// bit 15 = marker; bits 11:10 = matrix; bits 7:5 = row; bits 2:0 = column
/// ```
#[inline(always)]
const fn matrix_element(matrix: usize, row: usize, column: usize) -> u16 {
    0x8000 | ((matrix as u16) << 10) | ((row as u16) << 5) | column as u16
}

/// Pack adjacent b16 columns as `ldmatrix` returns them: lower column in bits
/// 0..15, upper column in bits 16..31.
#[inline(always)]
const fn matrix_word(matrix: usize, row: usize, pair: usize) -> u32 {
    let lower = matrix_element(matrix, row, pair * 2) as u32;
    let upper = matrix_element(matrix, row, pair * 2 + 1) as u32;
    lower | (upper << 16)
}

/// The same pair after a transposing load: the fragment's row/column pair maps
/// back to the original matrix's column pair/row.
#[inline(always)]
const fn matrix_word_transposed(matrix: usize, row: usize, pair: usize) -> u32 {
    let lower = matrix_element(matrix, pair * 2, row) as u32;
    let upper = matrix_element(matrix, pair * 2 + 1, row) as u32;
    lower | (upper << 16)
}

#[cuda_module]
mod kernels {
    use super::*;

    /// Write matrix `matrix`'s row `row` into shared row `row_index`.
    #[inline(always)]
    unsafe fn write_row(shared: *mut u32, row_index: usize, matrix: usize, row: usize) {
        let base = row_index * WORDS_PER_ROW;
        let mut pair = 0;
        while pair < WORDS_PER_ROW {
            unsafe {
                shared.add(base + pair).write(matrix_word(matrix, row, pair));
            }
            pair += 1;
        }
    }

    /// Initialize this lane's row and return both addresses for it: the
    /// per-lane pointer the element form uses, and the `.shared::cta.u32`
    /// address built by converting the shared base **once** and advancing it
    /// with byte arithmetic.
    #[inline(always)]
    unsafe fn row_addresses(shared: *mut u32, lane: usize, row_offset: u32) -> (*const u32, u32) {
        // Lane `n` owns row `n % 8` of matrix `n / 8`, so every lane holds a
        // legal 16-byte-aligned address even where x1/x2 only consume the
        // first eight or sixteen.
        let row_index = row_offset as usize + lane;
        unsafe {
            write_row(shared, row_index, lane / ROWS, lane % ROWS);
        }
        // `ldmatrix` is a weak read; this orders the writes above against the
        // warp's load.
        thread::sync_threads();

        let pointer = unsafe { shared.add(row_index * WORDS_PER_ROW).cast_const() };
        let native =
            unsafe { cvta_generic_to_shared_u32(shared.cast::<u8>()) }
                .wrapping_add(row_byte_offset(row_index));
        (pointer, native)
    }

    /// Pointer form in the first `N` register slots of the output, native form
    /// in the next `N`, one lane per column.
    #[inline(always)]
    unsafe fn store_pair(
        output: &mut DisjointSlice<u32>,
        lane: usize,
        via_pointer: &[u32],
        via_native: &[u32],
    ) {
        let registers = via_pointer.len();
        let mut index = 0;
        while index < registers {
            unsafe {
                *output.get_unchecked_mut(index * LANES + lane) = via_pointer[index];
                *output.get_unchecked_mut((registers + index) * LANES + lane) = via_native[index];
            }
            index += 1;
        }
    }

    // SAFETY for every kernel below: the launch is exactly one full 32-lane
    // warp, no lane exits, all lanes execute both loads unconditionally, and
    // both addresses name the same live, initialized, 16-byte-aligned row.

    #[kernel]
    pub fn x1_ab(mut output: DisjointSlice<u32>, row_offset: u32) {
        static mut INPUT: SharedArray<u32, SHARED_WORDS, 16> = SharedArray::UNINIT;
        let lane = thread_idx_x() as usize;
        if lane >= LANES {
            return;
        }
        let shared = core::ptr::addr_of_mut!(INPUT) as *mut u32;
        let (pointer_address, native_address) = unsafe { row_addresses(shared, lane, row_offset) };
        let via_pointer = unsafe { ldmatrix_x1(pointer_address) };
        let via_native = unsafe { ldmatrix_x1_shared_u32(native_address) };
        unsafe { store_pair(&mut output, lane, &[via_pointer], &[via_native]) };
    }

    #[kernel]
    pub fn x1_trans_ab(mut output: DisjointSlice<u32>, row_offset: u32) {
        static mut INPUT: SharedArray<u32, SHARED_WORDS, 16> = SharedArray::UNINIT;
        let lane = thread_idx_x() as usize;
        if lane >= LANES {
            return;
        }
        let shared = core::ptr::addr_of_mut!(INPUT) as *mut u32;
        let (pointer_address, native_address) = unsafe { row_addresses(shared, lane, row_offset) };
        let via_pointer = unsafe { ldmatrix_x1_trans(pointer_address) };
        let via_native = unsafe { ldmatrix_x1_trans_shared_u32(native_address) };
        unsafe { store_pair(&mut output, lane, &[via_pointer], &[via_native]) };
    }

    #[kernel]
    pub fn x2_ab(mut output: DisjointSlice<u32>, row_offset: u32) {
        static mut INPUT: SharedArray<u32, SHARED_WORDS, 16> = SharedArray::UNINIT;
        let lane = thread_idx_x() as usize;
        if lane >= LANES {
            return;
        }
        let shared = core::ptr::addr_of_mut!(INPUT) as *mut u32;
        let (pointer_address, native_address) = unsafe { row_addresses(shared, lane, row_offset) };
        let via_pointer = unsafe { ldmatrix_x2(pointer_address) };
        let via_native = unsafe { ldmatrix_x2_shared_u32(native_address) };
        unsafe { store_pair(&mut output, lane, &via_pointer, &via_native) };
    }

    #[kernel]
    pub fn x2_trans_ab(mut output: DisjointSlice<u32>, row_offset: u32) {
        static mut INPUT: SharedArray<u32, SHARED_WORDS, 16> = SharedArray::UNINIT;
        let lane = thread_idx_x() as usize;
        if lane >= LANES {
            return;
        }
        let shared = core::ptr::addr_of_mut!(INPUT) as *mut u32;
        let (pointer_address, native_address) = unsafe { row_addresses(shared, lane, row_offset) };
        let via_pointer = unsafe { ldmatrix_x2_trans(pointer_address) };
        let via_native = unsafe { ldmatrix_x2_trans_shared_u32(native_address) };
        unsafe { store_pair(&mut output, lane, &via_pointer, &via_native) };
    }

    #[kernel]
    pub fn x4_ab(mut output: DisjointSlice<u32>, row_offset: u32) {
        static mut INPUT: SharedArray<u32, SHARED_WORDS, 16> = SharedArray::UNINIT;
        let lane = thread_idx_x() as usize;
        if lane >= LANES {
            return;
        }
        let shared = core::ptr::addr_of_mut!(INPUT) as *mut u32;
        let (pointer_address, native_address) = unsafe { row_addresses(shared, lane, row_offset) };
        let via_pointer = unsafe { ldmatrix_x4(pointer_address) };
        let via_native = unsafe { ldmatrix_x4_shared_u32(native_address) };
        unsafe { store_pair(&mut output, lane, &via_pointer, &via_native) };
    }

    #[kernel]
    pub fn x4_trans_ab(mut output: DisjointSlice<u32>, row_offset: u32) {
        static mut INPUT: SharedArray<u32, SHARED_WORDS, 16> = SharedArray::UNINIT;
        let lane = thread_idx_x() as usize;
        if lane >= LANES {
            return;
        }
        let shared = core::ptr::addr_of_mut!(INPUT) as *mut u32;
        let (pointer_address, native_address) = unsafe { row_addresses(shared, lane, row_offset) };
        let via_pointer = unsafe { ldmatrix_x4_trans(pointer_address) };
        let via_native = unsafe { ldmatrix_x4_trans_shared_u32(native_address) };
        unsafe { store_pair(&mut output, lane, &via_pointer, &via_native) };
    }
}

// =============================================================================
// HOST
// =============================================================================

/// One shape's expected fragment for `lane`, register `index`.
fn expected(transposed: bool, index: usize, lane: usize) -> u32 {
    let row = lane / 4;
    let pair = lane % 4;
    if transposed {
        matrix_word_transposed(index, row, pair)
    } else {
        matrix_word(index, row, pair)
    }
}

macro_rules! check_case {
    ($module:expr, $stream:expr, $cfg:expr, $failed:ident, $name:expr, $transposed:expr, $registers:expr, $method:ident, $offsets:expr) => {{
        for offset in $offsets {
            let words = LANES * $registers * 2;
            let mut output = DeviceBuffer::<u32>::zeroed(&$stream, words)
                .expect("failed to allocate output");
            // SAFETY: one full warp, every lane in range, and the output covers
            // both halves (`registers * 2` words per lane).
            unsafe {
                $module.$method($stream.as_ref(), $cfg, &mut output, offset)
            }
            .expect("launch failed");
            let got = output.to_host_vec(&$stream).expect("copy failed");

            let mut agree = true;
            let mut matches_host = true;
            for lane in 0..LANES {
                for index in 0..$registers {
                    let pointer = got[index * LANES + lane];
                    let native = got[($registers + index) * LANES + lane];
                    let want = expected($transposed, index, lane);
                    if pointer != native {
                        agree = false;
                    }
                    if pointer != want || native != want {
                        matches_host = false;
                    }
                }
            }
            if !agree || !matches_host {
                $failed = true;
            }
            println!(
                "| {} | {} | {} | {} |",
                $name,
                offset,
                if agree { "equal" } else { "DIFFER" },
                if matches_host { "match" } else { "MISMATCH" }
            );
        }
    }};
}

fn main() {
    let context = CudaContext::new(0).expect("failed to create CUDA context");
    let (major, minor) = context
        .compute_capability()
        .expect("failed to query compute capability");
    if major * 10 + minor < 75 {
        println!("PASS (skipped): ldmatrix requires sm_75+; device is sm_{major}{minor}");
        return;
    }

    let stream = context.default_stream();
    let module = kernels::load(&context).expect("failed to load ldmatrix PTX");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (LANES as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    // Every case in the coverage table, at the base row and at non-zero offsets
    // that still leave the whole x4 tile inside the allocation.
    let offsets = [0u32, 8, MAX_ROW_OFFSET];

    let mut failed = false;
    println!("| shape | offset | pointer vs native | both vs host model |");
    println!("|---|---|---|---|");

    check_case!(module, stream, cfg, failed, "x1", false, 1, x1_ab, offsets);
    check_case!(module, stream, cfg, failed, "x1.trans", true, 1, x1_trans_ab, offsets);
    check_case!(module, stream, cfg, failed, "x2", false, 2, x2_ab, offsets);
    check_case!(module, stream, cfg, failed, "x2.trans", true, 2, x2_trans_ab, offsets);
    check_case!(module, stream, cfg, failed, "x4", false, 4, x4_ab, offsets);
    check_case!(module, stream, cfg, failed, "x4.trans", true, 4, x4_trans_ab, offsets);

    if failed {
        std::process::exit(1);
    }
    println!();
    println!("SUCCESS: both address forms return identical fragments for every shape");
}
