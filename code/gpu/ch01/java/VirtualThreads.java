// Java virtual threads (JEP 444, final in JDK 21): concurrency vs parallelism.
//
//   java VirtualThreads.java          (JDK 21+; single-file source launch, no javac step)
//
// Part 1: 10,000 tasks that each block for 100 ms (stand-in for I/O).
//         A pool of 200 platform (OS) threads: ~10,000/200 x 100 ms = ~5 s.
//         One virtual thread per task: ~100 ms. A blocked virtual thread is
//         unmounted from its carrier (an OS thread in a ForkJoinPool), so the
//         carrier runs someone else. Throughput for waiting, not speed.
// Part 2: CPU-bound tasks. Virtual threads give NO speedup over a platform pool
//         of NumCPU threads: they are multiplexed onto ~NumCPU carriers anyway.
// Part 3: blocking INSIDE `synchronized`. Before JDK 24 this pinned the virtual
//         thread to its carrier (JEP 444); JEP 491 (JDK 24) removed that, so on
//         JDK 24+ Part 3 takes ~100 ms, like Part 1. On JDK 21 at most ~NumCPU
//         tasks can be asleep at once, so 2,000 tasks take ~(2,000/NumCPU) x 100 ms.
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

public class VirtualThreads {
    static final int TASKS = 10_000;
    static final int PIN_TASKS = 2_000;  // smaller: on JDK 21 this part takes ~PIN_TASKS/NumCPU x 100 ms
    // How many tasks are inside their sleep at the same moment -- the direct
    // evidence of pinning (it is capped at the number of carrier threads).
    static final AtomicInteger sleeping = new AtomicInteger(), maxSleeping = new AtomicInteger();

    static long ms(long t0) { return (System.nanoTime() - t0) / 1_000_000; }

    static void sleep100() {
        maxSleeping.accumulateAndGet(sleeping.incrementAndGet(), Math::max);
        try { Thread.sleep(100); } catch (InterruptedException e) { throw new RuntimeException(e); }
        sleeping.decrementAndGet();
    }

    // Pure computation, no memory traffic; returns a checksum we can verify.
    static long work(long lo, long hi) {
        long acc = 0;
        for (long i = lo; i < hi; i++) {
            long x = i;
            for (int r = 0; r < 16; r++) { x ^= x >>> 33; x *= 0xff51afd7ed558ccdL; }
            acc += x;
        }
        return acc;
    }

    static long runBlocking(ExecutorService ex, boolean inSynchronized, int tasks) throws Exception {
        maxSleeping.set(0);
        long t0 = System.nanoTime();
        List<Future<?>> fs = new ArrayList<>();
        for (int i = 0; i < tasks; i++) {
            fs.add(ex.submit(() -> {
                if (inSynchronized) {
                    Object lock = new Object();  // private lock: no contention, only pinning
                    synchronized (lock) { sleep100(); }
                } else {
                    sleep100();
                }
            }));
        }
        for (Future<?> f : fs) f.get();
        return ms(t0);
    }

    static long[] runCpu(ExecutorService ex, long n, int chunks) throws Exception {
        long t0 = System.nanoTime();
        List<Future<Long>> fs = new ArrayList<>();
        for (int c = 0; c < chunks; c++) {
            long lo = n * c / chunks, hi = n * (c + 1) / chunks;
            fs.add(ex.submit(() -> work(lo, hi)));
        }
        long sum = 0;
        for (Future<Long> f : fs) sum += f.get();  // long + wraps, associative
        return new long[] {ms(t0), sum};
    }

    public static void main(String[] args) throws Exception {
        int ncpu = Runtime.getRuntime().availableProcessors();
        System.out.printf("Java %s, availableProcessors=%d%n", Runtime.version(), ncpu);
        Thread.ofVirtual().start(() ->
            System.out.println("a virtual thread prints itself as: " + Thread.currentThread())).join();

        System.out.printf("%nPart 1: %,d tasks x Thread.sleep(100 ms)%n", TASKS);
        try (ExecutorService pool = Executors.newFixedThreadPool(200)) {
            System.out.printf("  200 platform threads     : %6d ms%n", runBlocking(pool, false, TASKS));
        }
        try (ExecutorService vt = Executors.newVirtualThreadPerTaskExecutor()) {
            System.out.printf("  one virtual thread each  : %6d ms%n", runBlocking(vt, false, TASKS));
        }

        long n = 200_000_000L;
        long ref = work(0, n);           // warm-up run so the JIT has compiled work()
        long t0 = System.nanoTime();
        if (work(0, n) != ref) throw new AssertionError();
        long tSerial = ms(t0);
        System.out.printf("%nPart 2: CPU-bound hash of %,d longs, serial %d ms%n", n, tSerial);
        for (int warm = 0; warm < 2; warm++) {  // first pass warms up the JIT
            try (ExecutorService pool = Executors.newFixedThreadPool(ncpu);
                 ExecutorService vt = Executors.newVirtualThreadPerTaskExecutor()) {
                long[] p = runCpu(pool, n, 4 * ncpu), v = runCpu(vt, n, 4 * ncpu);
                if (warm == 1) {
                    System.out.printf("  %d platform threads      : %6d ms (%.1fx) %s%n", ncpu, p[0],
                        (double) tSerial / p[0], p[1] == ref ? "" : "WRONG");
                    System.out.printf("  virtual threads          : %6d ms (%.1fx) %s%n", v[0],
                        (double) tSerial / v[0], v[1] == ref ? "" : "WRONG");
                }
            }
        }

        System.out.printf("%nPart 3: %,d virtual threads, each sleeps 100 ms INSIDE synchronized%n", PIN_TASKS);
        try (ExecutorService vt = Executors.newVirtualThreadPerTaskExecutor()) {
            long t = runBlocking(vt, true, PIN_TASKS);
            String why = Runtime.version().feature() >= 24
                ? "JDK 24+: synchronized no longer pins (JEP 491)"
                : "JDK < 24: each sleeper PINS its carrier thread";
            System.out.printf("  %6d ms, max sleeping at once = %d   (%s)%n", t, maxSleeping.get(), why);
        }
    }
}
