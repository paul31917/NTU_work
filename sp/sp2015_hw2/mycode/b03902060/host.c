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
#include <sys/stat.h>
#include <time.h>
#include <fcntl.h>
typedef struct 
{
	int id;
	char index;
	int random_key;
	int money;
	int point;
	int rank;
}Player;
int main(int argc,char *argv[])
{
	int host_id=atoi(argv[1]);
	char FIFOname[5][30];
	sprintf(FIFOname[4],"host%d.FIFO",host_id);
	mkfifo(FIFOname[4],0666);
	sprintf(FIFOname[0],"host%d_A.FIFO",host_id);
	mkfifo(FIFOname[0],0666);
	sprintf(FIFOname[1],"host%d_B.FIFO",host_id);
	mkfifo(FIFOname[1],0666);
	sprintf(FIFOname[2],"host%d_C.FIFO",host_id);
	mkfifo(FIFOname[2],0666);
	sprintf(FIFOname[3],"host%d_D.FIFO",host_id);
	mkfifo(FIFOname[3],0666);
	int random;
	Player *players=malloc(sizeof(Player)*4);
 	srand(time(NULL));
	while(1)
	{
		scanf("%d%d%d%d",&players[0].id,&players[1].id,&players[2].id,&players[3].id);
		for(int i=0;i<4;i++)
		{
			players[i].index='A'+i;
			players[i].random_key=rand()%65536;
			players[i].point=0;
			players[i].rank=1;
		}
		if(players[0].id==-1)
			break;
		for(int i=0;i<4;i++)
		{
			pid_t pid;
			pid=fork();
			if(pid==0)
			{
				char host[30];
				char index[30];
				char key[30];
				sprintf(host,"%d",host_id);
				sprintf(index,"%c",players[i].index);
				sprintf(key,"%d",players[i].random_key);
				execlp("./player","./player",host,index,key,NULL);
			}
		}
		int fd[5];
		fd[4]=open(FIFOname[4],O_RDWR);
		for(int i=0;i<4;i++)
			fd[i]=open(FIFOname[i],O_WRONLY);
		for(int r=1;r<=11;r++)
		{
			char command[100];
			if(r==1)
			{
				sprintf(command,"1000 1000 1000 1000\n");
				for(int i=0;i<4;i++)
					players[i].money=1000;
			}
			else
			{
				FILE *fp=fdopen(fd[4],"r");
				Player *fourplayer=malloc(sizeof(Player)*4);
				for(int i=0;i<4;i++)
				{
					char callmoney[100];
					fgets(callmoney,100,fp);
					sscanf(callmoney,"%c%d%d",&fourplayer[i].index,&fourplayer[i].random_key,&fourplayer[i].money);
				}				
				for(int i=0;i<4;i++)
					players[i].money+=1000;
				for(int i=0;i<4;i++)
				{
					if(fourplayer[i].money!=0)
					{
						players[fourplayer[i].index-'A'].money-=fourplayer[i].money;
						players[fourplayer[i].index-'A'].point++;
						break;
					}
				}
				sprintf(command,"%d %d %d %d\n",players[0].money,players[1].money,players[2].money,players[3].money);
			}
			if(r!=11)
			{
				for(int i=0;i<4;i++)
					write(fd[i],command,strlen(command));
			}
		}
		for(int i=0;i<4;i++)
			for(int j=0;j<4;j++)
			{
				if(i!=j)
				{
					if(players[j].point>players[i].point)
						players[i].rank++;
				}
			}
		for(int i=0;i<4;i++)
		{
			printf("%d %d\n",players[i].id,players[i].rank);	
			close(fd[i]);
		}
		fflush(stdout);
		for(int i=0;i<4;i++)
		wait(NULL);
	}
	
	for(int i=0;i<5;i++)
		unlink(FIFOname[i]);
}













