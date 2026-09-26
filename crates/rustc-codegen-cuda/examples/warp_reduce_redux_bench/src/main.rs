/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

//! Paired measurement of `warp_reduce`'s two forms (#811 / #1305).
//!
//! One source, two builds: the fast path is selected by the
//! `cuda_oxide_sm_at_least` cfg the device build script derives from the target,
//! so there is no run-time switch to hold constant. That cfg is what #1305
//! adds; on a tree without it both builds take the butterfly and the harness
//! still runs, it just has nothing to compare.
//!
//! ```text
//!   cargo oxide run warp_reduce_redux_bench --arch sm_80   redux.sync
//!   cargo oxide run warp_reduce_redux_bench --arch sm_75   shuffle butterfly
//! ```
//!
//! Two timings per build, because they answer different questions:
//!
//!   * `*_chain` is a dependency chain — each reduction consumes the previous
//!     one — so it measures the latency of one reduction, which is what a
//!     serial scan or a scan-like recurrence pays.
//!   * `*_independent` runs four unrelated reductions per iteration, so the
//!     issue slots and shuffle port have something to overlap. A form that only
//!     wins in the chain is a latency win, not a throughput win.
//!
//! `shuffle_*` is the same butterfly `warp_reduce` uses when the fast path does
//! not apply, written out with `WarpCollective::shfl_xor`. Building at sm_80 and
//! comparing `api_chain` against `shuffle_chain` isolates the reduction from the
//! rest of the build, because both walk the same target and the same compiler.
//!
//! The reported figure is an amortised cost per warp reduction over a long
//! kernel, measured with a host wall clock around launch plus stream
//! synchronisation. That keeps launch and host overhead small per reduction but
//! does not remove them, and one warp reduction is one collective - not one
//! instruction, and not 32 independent results. The timing is checked to grow
//! with the iteration count, and every arm at every checked iteration count is
//! compared against a per-warp host simulation, so a loop the compiler folded
//! away cannot pass as a measurement.

use cuda_device::cooperative_groups::{
    WarpCollective, WarpTile, ops::Sum, this_thread_block, warp_reduce,
};
use cuda_device::{DisjointSlice, kernel, thread};
use cuda_host::cuda_module;

// =============================================================================
// KERNELS
// =============================================================================
#[cuda_module]
mod kernels {
    use super::*;

    /// The shuffle form of an unsigned sum, written out so it can be timed next
    /// to the API on the same target.
    #[inline(always)]
    fn butterfly_sum(warp: &WarpTile<32>, value: u32) -> u32 {
        let mut acc = value;
        let mut delta: u32 = 16;
        while delta > 0 {
            acc = acc.wrapping_add(warp.shfl_xor(acc, delta));
            delta >>= 1;
        }
        acc
    }

    #[inline(always)]
    fn butterfly_xor(warp: &WarpTile<32>, value: u32) -> u32 {
        let mut acc = value;
        let mut delta: u32 = 16;
        while delta > 0 {
            acc ^= warp.shfl_xor(acc, delta);
            delta >>= 1;
        }
        acc
    }

    #[inline(always)]
    fn butterfly_max(warp: &WarpTile<32>, value: u32) -> u32 {
        let mut acc = value;
        let mut delta: u32 = 16;
        while delta > 0 {
            let other = warp.shfl_xor(acc, delta);
            if other > acc {
                acc = other;
            }
            delta >>= 1;
        }
        acc
    }

    #[inline(always)]
    fn butterfly_min(warp: &WarpTile<32>, value: u32) -> u32 {
        let mut acc = value;
        let mut delta: u32 = 16;
        while delta > 0 {
            let other = warp.shfl_xor(acc, delta);
            if other < acc {
                acc = other;
            }
            delta >>= 1;
        }
        acc
    }

    /// Full-warp sum through the public API, chained.
    #[kernel]
    pub fn api_chain(input: &[u32], iterations: u32, mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let i = thread::index_1d().get() as usize;
        let mut v = input[i];
        let mut n = 0;
        while n < iterations {
            v = warp_reduce::<u32, Sum, _>(&warp, v);
            n += 1;
        }
        unsafe {
            *out.get_unchecked_mut(i) = v;
        }
    }

    /// The same chain with the butterfly spelled out.
    #[kernel]
    pub fn shuffle_chain(input: &[u32], iterations: u32, mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let i = thread::index_1d().get() as usize;
        let mut v = input[i];
        let mut n = 0;
        while n < iterations {
            v = butterfly_sum(&warp, v);
            n += 1;
        }
        unsafe {
            *out.get_unchecked_mut(i) = v;
        }
    }

    /// Four unrelated reductions per iteration, so latency can be hidden.
    #[kernel]
    pub fn api_independent(input: &[u32], iterations: u32, mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let i = thread::index_1d().get() as usize;
        let seed = input[i];
        let (mut a, mut b, mut c, mut d) = (
            seed,
            seed ^ 0x9e37_79b9,
            seed.wrapping_mul(3),
            seed.wrapping_add(0x85eb_ca6b),
        );
        let mut n = 0;
        while n < iterations {
            a = warp_reduce::<u32, Sum, _>(&warp, a);
            b = warp_reduce::<u32, cuda_device::cooperative_groups::ops::BitXor, _>(&warp, b);
            c = warp_reduce::<u32, cuda_device::cooperative_groups::ops::Max, _>(&warp, c);
            d = warp_reduce::<u32, cuda_device::cooperative_groups::ops::Min, _>(&warp, d);
            n += 1;
        }
        let base = i * 4;
        unsafe {
            *out.get_unchecked_mut(base) = a;
            *out.get_unchecked_mut(base + 1) = b;
            *out.get_unchecked_mut(base + 2) = c;
            *out.get_unchecked_mut(base + 3) = d;
        }
    }

    /// The same four chains over the butterfly.
    #[kernel]
    pub fn shuffle_independent(input: &[u32], iterations: u32, mut out: DisjointSlice<u32>) {
        let warp = this_thread_block().tiled_partition::<32>();
        let i = thread::index_1d().get() as usize;
        let seed = input[i];
        let (mut a, mut b, mut c, mut d) = (
            seed,
            seed ^ 0x9e37_79b9,
            seed.wrapping_mul(3),
            seed.wrapping_add(0x85eb_ca6b),
        );
        let mut n = 0;
        while n < iterations {
            a = butterfly_sum(&warp, a);
            b = butterfly_xor(&warp, b);
            c = butterfly_max(&warp, c);
            d = butterfly_min(&warp, d);
            n += 1;
        }
        let base = i * 4;
        unsafe {
            *out.get_unchecked_mut(base) = a;
            *out.get_unchecked_mut(base + 1) = b;
            *out.get_unchecked_mut(base + 2) = c;
            *out.get_unchecked_mut(base + 3) = d;
        }
    }
}

// =============================================================================
// HOST
// =============================================================================

const BLOCK: u32 = 128;
const WARPS: usize = (BLOCK / 32) as usize;

/// Iterations of the timed loop.
const TIMED_ITERATIONS: u32 = 40_000;

/// Iteration counts the correctness pass walks. Small enough to simulate, and
/// they include the degenerate ends: zero, one, and enough rounds for the sum
/// recurrence to reach its fixed point.
const CHECK_ITERATIONS: [u32; 6] = [0, 1, 2, 3, 8, 17];

/// Iteration counts for the scaling sanity check.
const SCALING_ITERATIONS: [u32; 4] = [1_000, 5_000, 20_000, 40_000];

const SAMPLES: usize = 15;
const WARMUPS: usize = 3;

/// Deterministic per-lane seeds, covering zero, all ones, the top bit and
/// wrapping extremes alongside ordinary values.
fn input(len: usize) -> Vec<u32> {
    let mut state = 0x2468_ace0u32;
    (0..len)
        .map(|i| match i % 8 {
            0 => 0,
            1 => u32::MAX,
            2 => 0x8000_0000,
            3 => 0x7fff_ffff,
            4 => 1,
            _ => {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                state
            }
        })
        .collect()
}

/// One iteration of the chain arms: every lane ends holding its warp's wrapping
/// sum, so the next round reduces 32 copies of it.
fn step_chain(values: &mut [u32]) {
    let mut next = values.to_vec();
    for warp in 0..WARPS {
        let base = warp * 32;
        let sum = values[base..base + 32]
            .iter()
            .fold(0u32, |acc, x| acc.wrapping_add(*x));
        next[base..base + 32].fill(sum);
    }
    values.copy_from_slice(&next);
}

/// One iteration of the independent arms. Every reduction reads the *previous*
/// round's whole warp and only then broadcasts, so no lane observes a value
/// another lane already updated this round.
fn step_independent(a: &mut [u32], b: &mut [u32], c: &mut [u32], d: &mut [u32]) {
    let mut next = (a.to_vec(), b.to_vec(), c.to_vec(), d.to_vec());
    for warp in 0..WARPS {
        let range = warp * 32..warp * 32 + 32;
        let sa = a[range.clone()]
            .iter()
            .fold(0u32, |acc, x| acc.wrapping_add(*x));
        let sb = b[range.clone()].iter().fold(0u32, |acc, x| acc ^ x);
        let sc = *c[range.clone()].iter().max().unwrap();
        let sd = *d[range.clone()].iter().min().unwrap();
        next.0[range.clone()].fill(sa);
        next.1[range.clone()].fill(sb);
        next.2[range.clone()].fill(sc);
        next.3[range].fill(sd);
    }
    a.copy_from_slice(&next.0);
    b.copy_from_slice(&next.1);
    c.copy_from_slice(&next.2);
    d.copy_from_slice(&next.3);
}

fn initial_independent(seed: u32) -> (u32, u32, u32, u32) {
    (
        seed,
        seed ^ 0x9e37_79b9,
        seed.wrapping_mul(3),
        seed.wrapping_add(0x85eb_ca6b),
    )
}

fn expected_chain(seeds: &[u32], iterations: u32) -> Vec<u32> {
    let mut values = seeds.to_vec();
    for _ in 0..iterations {
        step_chain(&mut values);
    }
    values
}

/// Four values per lane, in the order the kernels store them.
fn expected_independent(seeds: &[u32], iterations: u32) -> Vec<u32> {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut c = Vec::new();
    let mut d = Vec::new();
    for seed in seeds {
        let (va, vb, vc, vd) = initial_independent(*seed);
        a.push(va);
        b.push(vb);
        c.push(vc);
        d.push(vd);
    }
    for _ in 0..iterations {
        step_independent(&mut a, &mut b, &mut c, &mut d);
    }
    let mut out = Vec::with_capacity(seeds.len() * 4);
    for lane in 0..seeds.len() {
        out.extend_from_slice(&[a[lane], b[lane], c[lane], d[lane]]);
    }
    out
}

fn invert(values: &[u32]) -> Vec<u32> {
    values.iter().map(|value| !value).collect()
}

fn median_ms(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

fn main() {
    use cuda_core::simt::LaunchConfig;
    use cuda_core::{CudaContext, DeviceBuffer};
    use std::time::Instant;

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let module = kernels::load(&ctx).expect("Failed to load module");

    let cfg = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let len = BLOCK as usize;
    let seeds = input(len);
    let input_dev = DeviceBuffer::from_host(&stream, &seeds).unwrap();

    // ---- correctness: every arm, every iteration count, every lane ----
    let mut failed = false;
    for iterations in CHECK_ITERATIONS {
        for (name, method, per_lane) in [
            ("api_chain", 0u8, 1usize),
            ("shuffle_chain", 1, 1),
            ("api_independent", 2, 4),
            ("shuffle_independent", 3, 4),
        ] {
            let want = if per_lane == 4 {
                expected_independent(&seeds, iterations)
            } else {
                expected_chain(&seeds, iterations)
            };

            // The arm starts from the bitwise inverse of the oracle, so a value
            // the kernel never writes cannot read back as a correct answer.
            let mut out_dev = DeviceBuffer::from_host(&stream, &invert(&want)).unwrap();
            // SAFETY: launch shape matches the buffers; every lane stays in range.
            let result = unsafe {
                match method {
                    0 => {
                        module.api_chain(stream.as_ref(), cfg, &input_dev, iterations, &mut out_dev)
                    }
                    1 => module.shuffle_chain(
                        stream.as_ref(),
                        cfg,
                        &input_dev,
                        iterations,
                        &mut out_dev,
                    ),
                    2 => module.api_independent(
                        stream.as_ref(),
                        cfg,
                        &input_dev,
                        iterations,
                        &mut out_dev,
                    ),
                    _ => module.shuffle_independent(
                        stream.as_ref(),
                        cfg,
                        &input_dev,
                        iterations,
                        &mut out_dev,
                    ),
                }
            };
            result.expect("launch failed");
            let got = out_dev.to_host_vec(&stream).unwrap();

            let mut mismatch = None;
            'lanes: for lane in 0..len {
                for field in 0..per_lane {
                    let index = lane * per_lane + field;
                    if got[index] != want[index] {
                        mismatch = Some((lane, field, want[index], got[index]));
                        break 'lanes;
                    }
                }
            }
            match mismatch {
                None => {}
                Some((lane, field, want_value, got_value)) => {
                    println!(
                        "✗ {name}: iterations={iterations} warp={} lane={lane} field={field} want={want_value} got={got_value}",
                        lane / 32
                    );
                    failed = true;
                }
            }
        }
    }
    if failed {
        println!("correctness: FAIL");
        std::process::exit(1);
    }
    println!(
        "correctness: all arms match a per-warp host oracle for iterations {CHECK_ITERATIONS:?}"
    );

    let mut sink = DeviceBuffer::<u32>::zeroed(&stream, len * 4).unwrap();
    let launch = |arm: u8, iterations: u32, out: &mut DeviceBuffer<u32>| {
        // SAFETY: every arm's buffers cover its own indices.
        unsafe {
            match arm {
                0 => module.api_chain(stream.as_ref(), cfg, &input_dev, iterations, out),
                1 => module.shuffle_chain(stream.as_ref(), cfg, &input_dev, iterations, out),
                2 => module.api_independent(stream.as_ref(), cfg, &input_dev, iterations, out),
                _ => module.shuffle_independent(stream.as_ref(), cfg, &input_dev, iterations, out),
            }
        }
        .expect("launch failed");
    };

    // Scaling sanity check: a timed loop the compiler removed would not grow
    // with the iteration count, so the trend is part of the evidence.
    println!();
    println!("| iterations | api_chain ms | shuffle_chain ms |");
    println!("|---|---|---|");
    let mut scaling = Vec::new();
    for iterations in SCALING_ITERATIONS {
        let mut row = Vec::new();
        for arm in [0u8, 1u8] {
            for _ in 0..WARMUPS {
                launch(arm, iterations, &mut sink);
            }
            stream.synchronize().unwrap();
            let start = Instant::now();
            for _ in 0..3 {
                launch(arm, iterations, &mut sink);
            }
            stream.synchronize().unwrap();
            row.push(start.elapsed().as_secs_f64() * 1e3 / 3.0);
        }
        scaling.push((iterations, row[0], row[1]));
        println!("| {iterations} | {:.3} | {:.3} |", row[0], row[1]);
    }
    if !scaling.windows(2).all(|pair| pair[1].1 > pair[0].1) {
        println!();
        println!(
            "WARNING: timing did not grow with the iteration count; the loop may not be the measured work"
        );
    }

    // ---- paired measurement, round-robin so drift lands on every arm ----
    let arms: [(&str, u8, u64); 4] = [
        ("api_chain", 0, 1),
        ("shuffle_chain", 1, 1),
        ("api_independent", 2, 4),
        ("shuffle_independent", 3, 4),
    ];
    for (_, arm, _) in arms {
        for _ in 0..WARMUPS {
            launch(arm, TIMED_ITERATIONS, &mut sink);
        }
    }
    stream.synchronize().unwrap();

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(SAMPLES); arms.len()];
    for _ in 0..SAMPLES {
        for (index, (_, arm, _)) in arms.iter().enumerate() {
            let start = Instant::now();
            launch(*arm, TIMED_ITERATIONS, &mut sink);
            stream.synchronize().unwrap();
            samples[index].push(start.elapsed().as_secs_f64() * 1e3);
        }
    }

    let total_warps = u64::from(TIMED_ITERATIONS) * WARPS as u64;
    println!();
    println!(
        "timer_kind=host_wall_clock_kernel_batch iterations={TIMED_ITERATIONS} grid_blocks=1 warps_per_block={WARPS}"
    );
    println!();
    println!("| arm | R | median kernel ms | amortized ns per warp reduction |");
    println!("|---|---|---|---|");
    for (index, (name, _, factors)) in arms.iter().enumerate() {
        let reductions = total_warps * factors;
        let median = median_ms(samples[index].clone());
        let ns = median * 1e6 / reductions as f64;
        println!("| {name} | {factors} | {median:.3} | {ns:.2} |");
    }

    println!();
    println!("Note: amortized_ns_per_warp_reduction is a collective cost, not");
    println!("instruction latency. One warp_reduce is one collective, produced by");
    println!("either one redux.sync or five shuffle-and-combine rounds, and the");
    println!("figures include loop and launch overhead amortized over a long kernel.");
    println!();
    println!("SUCCESS: warp_reduce_redux_bench finished");
}
