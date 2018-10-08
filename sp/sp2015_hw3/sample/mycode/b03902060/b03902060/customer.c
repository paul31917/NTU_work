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
FILE *fp;
int cnt[3]={0};
int serial[3]={0};
int totalcust=0;
typedef struct 
{
	float tim;
	int command;//0~2=send sig,3=check sig
	int num;
}Timespot;
int compare(const void* a,const void* b)
{
	if((  (*(Timespot*)a).tim-(*(Timespot*)b).tim)<0)
		return -1;
	else
		return 1;
}
void sig_usr(int signo) {
    if (signo == SIGINT)
    {
        serial[0]++;
        fprintf(fp, "finish 0 %d\n", cnt[0]);
    }
    else if (signo == SIGUSR1) {
        serial[1]++;
        fprintf(fp, "finish 1 %d\n", serial[1]);
    }
    else {
        serial[2]++;
        fprintf(fp, "finish 2 %d\n", serial[2]);
    }
    
}
int main(int argc, char const *argv[])
{
	if (argc != 2) return 0;
    FILE *read_fp = fopen(argv[1], "r");
    pid_t ppid = getppid();
    struct timespec remain;
    struct sigaction act;
    act.sa_flags = 0;
    act.sa_handler = sig_usr;
    sigaction(SIGINT, &act, NULL);
    sigaction(SIGUSR1, &act, NULL);
    sigaction(SIGUSR2, &act, NULL);
    Timespot alltime[1000];
    int i=0;
    int count[3]={0};
    while(fscanf(read_fp,"%d%f",&alltime[i].command,&alltime[i].tim)!=EOF)
	{
		totalcust++;
		count[alltime[i].command]++;
		if(alltime[i].command==0)
		{
			i++;
			continue;
		}
		else
		{
			if(alltime[i].command==1)
			{
				alltime[i+1].command=alltime[i].command+3;
				alltime[i+1].tim=alltime[i].tim+1;
				alltime[i+1].num=count[alltime[i].command];
				i+=2;
			}
			else if(alltime[i].command==2)
			{
				alltime[i+1].command=alltime[i].command+3;
				alltime[i+1].tim=alltime[i].tim+0.3;
				alltime[i+1].num=count[alltime[i].command];
				i+=2;
			}
		}
	}   
	qsort(alltime,i,sizeof(Timespot),compare);
	fp = fopen("customer_log", "w");
	for(int j=0;j<i;j++)
	{
		if(alltime[j].command<3) //send
		{
			fprintf(fp, "send %d %d\n", alltime[j].command, ++cnt[alltime[j].command]);
			if(alltime[j].command==0)
			{
				write(1,"ordinary\n",9);
			}
			else if(alltime[j].command==1)
			{
				kill(ppid,SIGUSR1);
			}
			else
			{
				kill(ppid,SIGUSR2);
			}
		}
		else //check
		{
			if(alltime[j].command==1+3)
			{
				if(alltime[j].num>serial[1])
				{
					fprintf(fp, "timeout 1 %d\n", alltime[j].num);
					exit(0);
				}
				//else
				//	fprintf(fp, "finish 1 %d\n", alltime[j].num);
			}
			else if(alltime[j].command==2+3)
			{
				if(alltime[j].num>serial[2])
				{
					fprintf(fp, "timeout 2 %d\n", alltime[j].num);
					exit(0);
				}
				//else
					//fprintf(fp, "finish 2 %d\n", alltime[j].num);
			}
		}
		if((i-j)!=1)
		{
			float needtosleep=alltime[j+1].tim-alltime[j].tim;
			remain = (struct timespec){(int)needtosleep, (needtosleep-(int)needtosleep)*1000000000};
			while (nanosleep(&remain, &remain));
		}
	}
	while(1)
	{
		if(serial[0]+serial[1]+serial[2]==totalcust)
    		exit(0);
		pause();
	}
	return 0;
}








