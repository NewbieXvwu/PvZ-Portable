/* SIGPROF sampling profiler for the headless simulator.
 *
 * This machine has no perf, no valgrind and no sudo, so the tick hot spots are
 * sampled with a setitimer/SIGPROF handler that records the interrupted program
 * counter into a preallocated buffer.  At exit the buffer is written as binary
 * uint64 offsets (module-relative), which scripts/prof_symbolize.py turns into a
 * flat profile with addr2line.  Only the PC is sampled: that is what a flat
 * profile needs, and it stays async-signal-safe (no backtrace(), no malloc).
 *
 *   cc -shared -fPIC -O2 -o /tmp/prof_sample.so scripts/prof_sample.c -ldl
 *   PVZ_PROF_OUT=/tmp/pcs.bin LD_PRELOAD=/tmp/prof_sample.so ./build/pvz-portable -env ...
 */
#define _GNU_SOURCE
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>

#define CAPACITY (1 << 22) /* 4M samples: 2 kHz for half an hour */

static uint64_t *g_samples;
static volatile size_t g_count;
static uintptr_t g_base; /* load bias of the main executable */

static void on_prof(int signal, siginfo_t *info, void *context)
{
	(void)signal; (void)info;
	if (g_count >= CAPACITY) return;
	ucontext_t *state = (ucontext_t *)context;
	uintptr_t pc = (uintptr_t)state->uc_mcontext.gregs[REG_RIP];
	g_samples[g_count++] = (uint64_t)(pc - g_base);
}

/* The main executable's first mapping with file offset 0 gives the PIE load bias. */
static uintptr_t executable_base(void)
{
	FILE *maps = fopen("/proc/self/maps", "r");
	if (!maps) return 0;
	char line[4096], path[4096];
	unsigned long start, end, offset;
	uintptr_t base = 0;
	while (fgets(line, sizeof line, maps))
	{
		if (sscanf(line, "%lx-%lx %*s %lx %*s %*s %4095s", &start, &end, &offset, path) != 4) continue;
		if (offset != 0 || path[0] != '/') continue;
		/* the executable is the first file mapping whose name is our own image */
		if (strstr(path, "pvz-portable") == NULL) continue;
		base = (uintptr_t)start;
		break;
	}
	fclose(maps);
	return base;
}

static void dump(void)
{
	const char *path = getenv("PVZ_PROF_OUT");
	if (!path) path = "/tmp/pvz_pcs.bin";
	FILE *out = fopen(path, "wb");
	if (!out) return;
	fwrite(g_samples, sizeof(uint64_t), g_count, out);
	fclose(out);
}

__attribute__((constructor)) static void start_profiler(void)
{
	g_base = executable_base();
	g_samples = (uint64_t *)malloc(CAPACITY * sizeof(uint64_t));
	if (!g_samples) return;
	struct sigaction action;
	memset(&action, 0, sizeof action);
	action.sa_sigaction = on_prof;
	action.sa_flags = SA_SIGINFO | SA_RESTART;
	sigemptyset(&action.sa_mask);
	if (sigaction(SIGPROF, &action, NULL) != 0) return;
	struct itimerval timer;
	timer.it_interval.tv_sec = 0;
	timer.it_interval.tv_usec = 500; /* 2 kHz */
	timer.it_value = timer.it_interval;
	setitimer(ITIMER_PROF, &timer, NULL);
	atexit(dump);
}
