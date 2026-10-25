/* SIGPROF sampling profiler for the headless simulator.
 *
 * This machine has no perf, no valgrind and no sudo, so the tick hot spots are sampled
 * with a setitimer/SIGPROF handler that records the interrupted program counter into a
 * preallocated buffer.  At exit the buffer is resolved to (module, offset) pairs using
 * dladdr -- so the 25% that lands in libc is not lost as "??" -- and written to
 * PVZ_PROF_OUT for scripts/prof_symbolize.py, which runs addr2line on the executable.
 *
 *   cc -shared -fPIC -O2 -o /tmp/prof_sample.so scripts/prof_sample.c -ldl
 *   PVZ_PROF_OUT=/tmp/pcs.bin LD_PRELOAD=/tmp/prof_sample.so ./build/pvz-portable -env ...
 *
 * Only the PC is sampled: that is what a flat profile needs and it stays
 * async-signal-safe (no backtrace(), no malloc, no dladdr inside the handler).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <ucontext.h>
#include <unistd.h>

#define CAPACITY (1 << 22) /* 4M samples: 2 kHz for half an hour */
#define MODULES 32

static uint64_t *g_raw;
static volatile size_t g_count;
static int g_found; /* only the simulator process profiles; LD_PRELOAD also hits python */
static volatile sig_atomic_t g_armed = 1;

/* PVZ_PROF_ARM_ONLY=1 starts disarmed: the caller then brackets the phase it cares about
 * (the rollouts, not the episode replay or the resource loading) with SIGUSR1/SIGUSR2. */
static void on_arm(int signal) { g_armed = signal == SIGUSR1; }

static void on_prof(int signal, siginfo_t *info, void *context)
{
	(void)signal; (void)info;
	if (!g_armed || g_count >= CAPACITY) return;
	ucontext_t *state = (ucontext_t *)context;
	g_raw[g_count++] = (uint64_t)state->uc_mcontext.gregs[REG_RIP];
}

static int is_simulator(void)
{
	FILE *maps = fopen("/proc/self/maps", "r");
	if (!maps) return 0;
	char line[4096], path[4096];
	unsigned long start, end, offset;
	int found = 0;
	while (fgets(line, sizeof line, maps))
		if (sscanf(line, "%lx-%lx %*s %lx %*s %*s %4095s", &start, &end, &offset, path) == 4 &&
			strstr(path, "pvz-portable") != NULL)
		{
			found = 1;
			break;
		}
	fclose(maps);
	return found;
}

static void dump(void)
{
	/* Freeze the count before allocating/writing the two parallel arrays. SIGPROF
	 * used to keep appending here, so their lengths could differ by one sample. */
	g_armed = 0;
	struct itimerval stopped = {0};
	setitimer(ITIMER_PROF, &stopped, NULL);
	const size_t sample_count = g_count;
	const char *path = getenv("PVZ_PROF_OUT");
	if (!path) path = "/tmp/pvz_pcs.bin";
	char *names[MODULES];
	memset(names, 0, sizeof names);
	uint64_t *offsets = (uint64_t *)malloc(sample_count * sizeof(uint64_t));
	uint32_t *modules = (uint32_t *)malloc(sample_count * sizeof(uint32_t));
	if (!offsets || !modules) return;
	int used = 0;
	for (size_t index = 0; index < sample_count; ++index)
	{
		Dl_info info;
		if (!dladdr((void *)(uintptr_t)g_raw[index], &info) || !info.dli_fbase)
		{
			offsets[index] = g_raw[index];
			modules[index] = 0xffffffffu;
			continue;
		}
		uint32_t slot = 0xffffffffu;
		for (int known = 0; known < used; ++known)
			if (names[known] && strcmp(names[known], info.dli_fname ? info.dli_fname : "?") == 0)
			{
				slot = (uint32_t)known;
				break;
			}
		if (slot == 0xffffffffu && used < MODULES)
		{
			slot = (uint32_t)used;
			names[used++] = strdup(info.dli_fname ? info.dli_fname : "?");
		}
		offsets[index] = (uint64_t)((uintptr_t)g_raw[index] - (uintptr_t)info.dli_fbase);
		modules[index] = slot;
	}
	FILE *out = fopen(path, "wb");
	if (!out) return;
	fprintf(out, "PVZPROF 1 %d\n", used);
	for (int index = 0; index < used; ++index) fprintf(out, "%s\n", names[index] ? names[index] : "?");
	fwrite(offsets, sizeof(uint64_t), sample_count, out);
	fwrite(modules, sizeof(uint32_t), sample_count, out);
	fclose(out);
}

__attribute__((constructor)) static void start_profiler(void)
{
	if (!is_simulator()) return;
	g_found = 1;
	g_raw = (uint64_t *)malloc(CAPACITY * sizeof(uint64_t));
	if (!g_raw) return;
	struct sigaction action;
	memset(&action, 0, sizeof action);
	action.sa_sigaction = on_prof;
	action.sa_flags = SA_SIGINFO | SA_RESTART;
	sigemptyset(&action.sa_mask);
	if (sigaction(SIGPROF, &action, NULL) != 0) return;
	struct sigaction arming;
	memset(&arming, 0, sizeof arming);
	arming.sa_handler = on_arm;
	arming.sa_flags = SA_RESTART;
	sigemptyset(&arming.sa_mask);
	sigaction(SIGUSR1, &arming, NULL);
	sigaction(SIGUSR2, &arming, NULL);
	if (getenv("PVZ_PROF_ARM_ONLY")) g_armed = 0;
	struct itimerval timer;
	timer.it_interval.tv_sec = 0;
	timer.it_interval.tv_usec = 500; /* 2 kHz */
	timer.it_value = timer.it_interval;
	setitimer(ITIMER_PROF, &timer, NULL);
	atexit(dump);
}
