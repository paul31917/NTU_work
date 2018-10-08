#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>

int cnt[3], current_time, timeout[3], done0;
FILE *fp;

int in_handler;
static void sig_usr(int signo) {
    if (signo == SIGINT)
        fprintf(fp, "finish 0 %d\n", ++done0);
    else if (signo == SIGUSR1) {
        fprintf(fp, "finish 1 %d\n", cnt[1]);
        timeout[1] = 0;
    } else {
        fprintf(fp, "finish 2 %d\n", cnt[2]);
        timeout[2] = 0;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) return 0;
    int prio, tim, i;
    FILE *read_fp = fopen(argv[1], "r");
    pid_t ppid = getppid();
    struct timespec remain;
    struct sigaction act;
    act.sa_flags = 0;
    act.sa_handler = sig_usr;
    sigemptyset(&act.sa_mask);
    sigaction(SIGINT, &act, NULL);
    sigaction(SIGUSR1, &act, NULL);
    sigaction(SIGUSR2, &act, NULL);

    fp = fopen("sender_log", "w");
    while (1) {
        fscanf(read_fp, "%d %d", &prio, &tim);
        if (feof(read_fp)) break;
        if (ferror(read_fp)) continue;
        for (; current_time < tim; ++current_time) {
            for (i = 1; i <= 2; ++i)
                if (timeout[i] != 0 && timeout[i] < current_time) {
                    fprintf(fp, "timeout %d %d\n", i, cnt[i]);
                    exit(0);
                }
            remain = (struct timespec){0, 100000000};
            while (nanosleep(&remain, &remain));
        }
        fprintf(fp, "send %d %d\n", prio, ++cnt[prio]);
        if (prio == 0) {
            printf("ordinary\n");
            fflush(stdout);
        }
        else if (prio == 1) {
            kill(ppid, SIGUSR1);
            timeout[1] = current_time + 10;
        } else {
            kill(ppid, SIGUSR2);
            timeout[2] = current_time + 2;
        }
    }
    for (;cnt[0] > done0 || current_time <= timeout[1] || current_time <= timeout[2]; ++current_time) {
        remain = (struct timespec){0, 100000000};
        while (nanosleep(&remain, &remain));
    }
    fclose(fp);
    return 0;
}
