#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <signal.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>

int cnt0, cnt1, cnt2;
pid_t pid;
FILE *fp;

static void sig_int(int signo) {
    if (signo != SIGINT) return;
    kill(pid, SIGKILL);
    wait(NULL);
    fprintf(fp, "terminate\n");
    exit(0);
}

static void sig_usr1(int signo) {
    if (signo != SIGUSR1) return;
    fprintf(fp, "receive 1 %d\n", ++cnt1);
    struct timespec remain = (struct timespec){0, 500000000};
    while (nanosleep(&remain, &remain));
    kill(pid, SIGUSR1);
    fprintf(fp, "finish 1 %d\n", cnt1);
}

static void sig_usr2(int signo) {
    if (signo != SIGUSR2) return;
    fprintf(fp, "receive 2 %d\n", ++cnt2);
    struct timespec remain = (struct timespec){0, 200000000};
    while (nanosleep(&remain, &remain));
    kill(pid, SIGUSR2);
    fprintf(fp, "finish 2 %d\n", cnt2);
}

int main(int argc, char *argv[]) {
    if (argc != 2) return 0;
    struct sigaction act;

    act.sa_flags = 0;
    act.sa_handler = sig_usr1;
    sigemptyset(&act.sa_mask);
    sigaction(SIGUSR1, &act, NULL);

    act.sa_handler = sig_usr2;
    sigaddset(&act.sa_mask, SIGUSR1);
    sigaction(SIGUSR2, &act, NULL);

    act.sa_handler = sig_int;
    sigaddset(&act.sa_mask, SIGUSR2);
    sigaction(SIGINT, &act, NULL);

    fp = fopen("receiver_log", "w");
    int pipe_fd[2];
    pipe(pipe_fd);
    if ((pid = fork()) == 0) {
        dup2(pipe_fd[1], 1);
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        execl("./sender", "sender", argv[1], NULL);
    }
    dup2(pipe_fd[0], 0);
    close(pipe_fd[0]);
    close(pipe_fd[1]);

    while (1) {
        scanf("%*s");
        if (feof(stdin)) break;
        if (ferror(stdin)) {
            clearerr(stdin);
            continue;
        }
        fprintf(fp, "receive 0 %d\n", ++cnt0);
        struct timespec remain = (struct timespec){1, 0};
        while (nanosleep(&remain, &remain));
        kill(pid, SIGINT);
        fprintf(fp, "finish 0 %d\n", cnt0);
    }
    wait(NULL);
    fprintf(fp, "terminate\n");
    fclose(fp);
    return 0;
}
