/* Experimental Linux x86-64 CPU profiler; no production binary/source changes.
 * gcc -O2 -shared -fPIC scripts/task6/cpu_pc_sample.c -o /tmp/pc_sample.so
 * mkdir -p /tmp/pc-profile
 * LD_PRELOAD=/tmp/pc_sample.so PVZ_PC_SAMPLE_DIR=/tmp/pc-profile <teacher command>
 * Only pvz-portable processes are sampled, once per10ms of process CPU time.
 * The signal handler only stores the saved instruction pointer; no allocation,
 * unwinding, formatting or I/O occurs in the handler. Profiles/maps write at exit.
 * Both ABBA arms must use the same sampler. Python CPU is not represented.
 */
#define _GNU_SOURCE
#include <stdint.h>
#include <signal.h>
#include <ucontext.h>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
static uintptr_t pcs[1<<20];
static volatile sig_atomic_t count;
static char output[4096];
static int active;
static void sample(int signal_number, siginfo_t *info, void *context) {
    (void)signal_number;(void)info;
    sig_atomic_t n=count;
    if(n<(1<<20)) { pcs[n]=(uintptr_t)((ucontext_t*)context)->uc_mcontext.gregs[REG_RIP]; count=n+1; }
}
static void write_all(int fd,const void *buffer,size_t bytes) {
    const char *p=buffer;
    while(bytes) {ssize_t n=write(fd,p,bytes);if(n<0&&errno==EINTR)continue;if(n<=0)break;p+=n;bytes-=n;}
}
__attribute__((constructor)) static void start(void) {
    const char *directory=getenv("PVZ_PC_SAMPLE_DIR");if(!directory||!*directory)return;
    char exe[4096];ssize_t size=readlink("/proc/self/exe",exe,sizeof(exe)-1);if(size<=0)return;exe[size]=0;
    const char *target=getenv("PVZ_PC_SAMPLE_EXE");if(!target)target="pvz-portable";
    const char *name=strrchr(exe,'/');name=name?name+1:exe;if(strcmp(name,target))return;
    snprintf(output,sizeof(output),"%s/%ld",directory,(long)getpid());
    struct sigaction action={0};action.sa_sigaction=sample;action.sa_flags=SA_SIGINFO|SA_RESTART;sigemptyset(&action.sa_mask);
    if(sigaction(SIGPROF,&action,0))return;
    struct itimerval timer={.it_interval={0,10000},.it_value={0,10000}};
    if(!setitimer(ITIMER_PROF,&timer,0))active=1;
}
__attribute__((destructor)) static void finish(void) {
    if(!active)return;
    struct itimerval timer={0};setitimer(ITIMER_PROF,&timer,0);
    char path[4200];snprintf(path,sizeof(path),"%s.pcs",output);int fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0600);
    if(fd>=0){write_all(fd,pcs,(size_t)count*sizeof(pcs[0]));close(fd);}
    snprintf(path,sizeof(path),"%s.maps",output);fd=open(path,O_WRONLY|O_CREAT|O_TRUNC,0600);int in=open("/proc/self/maps",O_RDONLY);
    if(fd>=0&&in>=0){char buffer[8192];ssize_t n;while((n=read(in,buffer,sizeof(buffer)))>0)write_all(fd,buffer,(size_t)n);}
    if(fd>=0)close(fd);
    if(in>=0)close(in);
}
