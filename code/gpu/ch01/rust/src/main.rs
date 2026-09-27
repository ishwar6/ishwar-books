//! Tokio: async concurrency is not CPU parallelism.
//!
//!   cargo run --release
//!
//! Part 1  10,000 async tasks that each await a 100 ms timer, on a runtime with
//!         ONE thread: ~100 ms total. An awaiting task is just a parked state
//!         machine; the thread runs whichever task is ready.
//! Part 2  CPU-bound chunks, four ways:
//!           (a) tokio::spawn on a current_thread runtime  -> no speedup (1 thread)
//!           (b) tokio::spawn on a multi_thread runtime    -> speedup (N workers,
//!               work-stealing) -- but it starves any I/O tasks on those workers
//!           (c) spawn_blocking                            -> runs on the separate
//!               blocking pool; works, but the docs say to bound it (semaphore)
//!           (d) rayon par_iter                            -> the tool made for this
//! Part 3  The classic bug: blocking the executor. std::thread::sleep inside async
//!         tasks on 2 worker threads serializes them; tokio::time::sleep does not.
use std::time::{Duration, Instant};

/// Pure computation, no memory traffic (same hash as the Go and Java demos).
fn work(lo: u64, hi: u64) -> u64 {
    let mut acc: u64 = 0;
    for i in lo..hi {
        let mut x = i;
        for _ in 0..16 {
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51afd7ed558ccd);
        }
        acc = acc.wrapping_add(x);
    }
    acc
}

fn chunks(n: u64, k: u64) -> Vec<(u64, u64)> {
    (0..k).map(|c| (n * c / k, n * (c + 1) / k)).collect()
}

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1e3
}

/// Spawn every chunk as an async task on `rt`, sum the results.
fn spawn_all(rt: &tokio::runtime::Runtime, parts: &[(u64, u64)]) -> u64 {
    rt.block_on(async {
        let hs: Vec<_> = parts.iter().map(|&(lo, hi)| tokio::spawn(async move { work(lo, hi) })).collect();
        let mut s = 0u64;
        for h in hs {
            s = s.wrapping_add(h.await.unwrap());
        }
        s
    })
}

fn main() {
    let ncpu = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("available_parallelism = {ncpu}\n");

    // ---------------- Part 1
    let one = tokio::runtime::Builder::new_current_thread().enable_time().build().unwrap();
    let t = Instant::now();
    one.block_on(async {
        let hs: Vec<_> = (0..10_000)
            .map(|_| tokio::spawn(async { tokio::time::sleep(Duration::from_millis(100)).await }))
            .collect();
        for h in hs {
            h.await.unwrap();
        }
    });
    println!("Part 1: 10,000 tasks x sleep(100 ms) on ONE thread: {:.1} ms\n", ms(t));

    // ---------------- Part 2
    let n: u64 = 200_000_000;
    let parts = chunks(n, 4 * ncpu as u64);
    let t = Instant::now();
    let reference = work(0, n);
    let serial = ms(t);
    println!("Part 2: CPU-bound hash of {n} integers, serial {serial:.0} ms");
    let check = |s: u64| if s == reference { "" } else { "  <-- WRONG" };

    let t = Instant::now();
    let s = spawn_all(&one, &parts);
    let d = ms(t);
    println!("  (a) tokio::spawn, current_thread     : {d:7.0} ms {:5.2}x{}", serial / d, check(s));

    let multi = tokio::runtime::Builder::new_multi_thread().worker_threads(ncpu).enable_time().build().unwrap();
    let t = Instant::now();
    let s = spawn_all(&multi, &parts);
    let d = ms(t);
    println!("  (b) tokio::spawn, multi_thread({ncpu:2})  : {d:7.0} ms {:5.2}x{}", serial / d, check(s));

    let t = Instant::now();
    let s = one.block_on(async {
        let hs: Vec<_> = parts.iter().map(|&(lo, hi)| tokio::task::spawn_blocking(move || work(lo, hi))).collect();
        let mut s = 0u64;
        for h in hs {
            s = s.wrapping_add(h.await.unwrap());
        }
        s
    });
    let d = ms(t);
    println!("  (c) spawn_blocking from current_thread: {d:7.0} ms {:5.2}x{}", serial / d, check(s));

    use rayon::prelude::*;
    let t = Instant::now();
    let s = parts.par_iter().map(|&(lo, hi)| work(lo, hi)).reduce(|| 0u64, |a, b| a.wrapping_add(b));
    let d = ms(t);
    println!("  (d) rayon par_iter                   : {d:7.0} ms {:5.2}x{}\n", serial / d, check(s));

    // ---------------- Part 3
    let two = tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_time().build().unwrap();
    for (label, blocking) in [("std::thread::sleep (blocks worker)", true), ("tokio::time::sleep (yields)       ", false)] {
        let t = Instant::now();
        two.block_on(async move {
            let hs: Vec<_> = (0..10)
                .map(|_| {
                    tokio::spawn(async move {
                        if blocking {
                            std::thread::sleep(Duration::from_millis(100)); // WRONG in async code
                        } else {
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    })
                })
                .collect();
            for h in hs {
                h.await.unwrap();
            }
        });
        println!("Part 3: 10 tasks, 2 workers, {label}: {:6.1} ms", ms(t));
    }
}
