#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <time.h>
int memserial=1;
int vipserial=1;
int ordserial=1;
pid_t pid;
FILE *fp;
static void sig_usr1(int signo) {
    fprintf(fp, "receive 1 %d\n", memserial);
    struct timespec remain = (struct timespec){0, 500000000};
    while (nanosleep(&remain, &remain));
    kill(pid, SIGUSR1);
    fprintf(fp, "finish 1 %d\n", memserial);
    memserial++;
}

static void sig_usr2(int signo) {
    fprintf(fp, "receive 2 %d\n", vipserial);
    struct timespec remain = (struct timespec){0, 200000000};
    while (nanosleep(&remain, &remain));
    kill(pid, SIGUSR2);
    fprintf(fp, "finish 2 %d\n", vipserial);
    vipserial++;
}
int main(int argc, char const *argv[])
{
	struct sigaction act;
    act.sa_flags = 0;

    act.sa_handler = sig_usr1;
    sigaddset(&act.sa_mask, SIGUSR1);
    sigaction(SIGUSR1, &act, NULL);

    act.sa_handler = sig_usr2;
    sigaddset(&act.sa_mask, SIGUSR1);
    sigaddset(&act.sa_mask, SIGUSR2);
    sigaction(SIGUSR2, &act, NULL);
	fp = fopen("bidding_system_log", "w");
	int pipe_fd[2];
    pipe(pipe_fd);
    if ((pid = fork()) == 0) {
        dup2(pipe_fd[1], 1);
        close(pipe_fd[0]);
        execl("./customer", "customer",argv[1], NULL);
    }
    close(pipe_fd[1]);
    char buffer[30];
    while (read(pipe_fd[0],buffer,30)>0) {
        fprintf(fp, "receive 0 %d\n", ordserial);
        struct timespec remain = (struct timespec){1, 0};
        while (nanosleep(&remain, &remain));
        kill(pid, SIGINT);
        fprintf(fp, "finish 0 %d\n", ordserial);
        ordserial++;
    }
    wait(NULL);
    fprintf(fp, "terminate\n");
    fclose(fp);
    return 0;
}