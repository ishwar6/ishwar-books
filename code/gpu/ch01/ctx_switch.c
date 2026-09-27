// ctx_switch.c -- what does it cost the OS to switch from one thread to another?
//
// Two threads ping-pong one byte through a pair of pipes. Each round trip is
// two blocking reads, i.e. (at least) two sleeps + wake-ups through the kernel
// scheduler. We report time per one-way hand-off. Compare with a GPU warp
// switch, which costs zero cycles because every resident warp's registers
// already live in the register file (§1.8).
//
// This is an upper-bound style measurement: it includes the pipe syscalls, not
// just the register save/restore. That is the honest number for "block on I/O
// and let another thread run", which is what the comparison is about.
//
//   cc -O2 ctx_switch.c -lpthread -o ctx_switch
#include "bench.h"

#include <pthread.h>
#include <unistd.h>

#define ROUNDS 200000

static int ping[2], pong[2];

static void* echo(void* arg) {
    (void)arg;
    char c;
    for (int i = 0; i < ROUNDS; ++i) {
        if (read(ping[0], &c, 1) != 1) exit(1);
        if (write(pong[1], &c, 1) != 1) exit(1);
    }
    return NULL;
}

int main(void) {
    if (pipe(ping) || pipe(pong)) return 1;
    pthread_t th;
    pthread_create(&th, NULL, echo, NULL);
    char c = 'x';
    double t0 = now_sec();
    for (int i = 0; i < ROUNDS; ++i) {
        if (write(ping[1], &c, 1) != 1) return 1;
        if (read(pong[0], &c, 1) != 1) return 1;
    }
    double dt = now_sec() - t0;
    pthread_join(th, NULL);
    printf("%d round trips in %.1f ms -> %.2f us per one-way thread hand-off\n", ROUNDS,
           dt * 1e3, dt / (2.0 * ROUNDS) * 1e6);
    return 0;
}
