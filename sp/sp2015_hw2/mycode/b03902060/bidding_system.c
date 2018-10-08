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
struct playersingame
{
	int A;
	int B;
	int C;
	int D;
};
void fillallplayer(struct playersingame *playeringame,int player_num)
{
	int game=0;
	for(int i=1;i<=player_num-3;i++)
		for(int j=i+1;j<=player_num-2;j++)
			for(int k=j+1;k<=player_num-1;k++)
				for(int l=k+1;l<=player_num;l++)
				{
					playeringame[game].A=i;
					playeringame[game].B=j;
					playeringame[game].C=k;
					playeringame[game].D=l;
					game++;
				}
	return;
}
int calgamenum(int num)
{
	return ((num*(num-1)/2)*(num-2)/3)*(num-3)/4;
}
int main(int argc, char const *argv[])
{
	int host_num=atoi(argv[1]);
	int player_num=atoi(argv[2]);
	int grade[player_num+1];
	for(int i=1;i<=player_num;i++)
		grade[i]=0;
	int gamenum=calgamenum(player_num);
	int resultnum=gamenum;
	struct playersingame *playeringame=malloc(sizeof(struct playersingame)*gamenum);
	fillallplayer(playeringame,player_num);
	int bidding_to_host[host_num+1][2];
	int host_to_bidding[host_num+1][2];
	int maxfd=-1;
	for(int i=1;i<=host_num;i++)
	{
		pipe(bidding_to_host[i]);
		pipe(host_to_bidding[i]);
		if(host_to_bidding[i][0]>maxfd)
			maxfd=host_to_bidding[i][0];
		pid_t pid;
		pid=fork();
		if(pid==0)
		{
			close(bidding_to_host[i][1]);
			dup2(bidding_to_host[i][0],0);
			close(host_to_bidding[i][0]);
			dup2(host_to_bidding[i][1],1);
			char hostid[20];
			sprintf(hostid,"%d",i);
			execlp("./host","./host",hostid,NULL);
		}
		else
		{
			close(bidding_to_host[i][0]);
			close(host_to_bidding[i][1]);
		}
	}
	int gamecount=0;
	for(int i=1;i<=host_num;i++)
	{
		if(gamecount<gamenum)
		{
			char task[30];
			sprintf(task,"%d %d %d %d\n",playeringame[gamecount].A,playeringame[gamecount].B,playeringame[gamecount].C,playeringame[gamecount].D);
			write(bidding_to_host[i][1],task,strlen(task));
			gamecount++;
		}
	}
	fd_set allhostset,cpyset;
	FD_ZERO(&allhostset);
	for(int i=1;i<=host_num;i++)
		FD_SET(host_to_bidding[i][0],&allhostset);
	while(1)
	{
		FD_ZERO(&cpyset);
		cpyset=allhostset;
		int success=select(maxfd+1,&cpyset,NULL,NULL,NULL);
		int working_fd;
		int working_host;
		for(int i=0;i<maxfd+1;i++)
		{
			if(FD_ISSET(i,&cpyset))
			{
				working_fd=i;
				for(int j=1;j<=host_num;j++)
				{
					if(working_fd==host_to_bidding[j][0])
					{
						working_host=j;
						break;
					}
				}
				char result[100];
				read(host_to_bidding[working_host][0],result,sizeof(result));
				resultnum--;
				int fourplayers[4];
				int fourlevels[4];
				sscanf(result,"%d%d%d%d%d%d%d%d",&fourplayers[0],&fourlevels[0],&fourplayers[1],&fourlevels[1],&fourplayers[2],&fourlevels[2],&fourplayers[3],&fourlevels[3]);
				for(int k=0;k<4;k++)
				{
					if(fourlevels[k]==1)
						grade[fourplayers[k]]+=3;
					else if(fourlevels[k]==2)
						grade[fourplayers[k]]+=2;
					else if(fourlevels[k]==3)
						grade[fourplayers[k]]+=1;
					else if(fourlevels[k]==4)
						grade[fourplayers[k]]+=0;
				}
				if(!(gamecount<gamenum))
				{
					continue;
				}	
				else
				{
					char task[30];
					sprintf(task,"%d %d %d %d\n",playeringame[gamecount].A,playeringame[gamecount].B,playeringame[gamecount].C,playeringame[gamecount].D);
					write(bidding_to_host[working_host][1],task,strlen(task));
					gamecount++;
				}
			}
		}
		if(resultnum==0)
		{
			for(int k=1;k<host_num;k++)
			{
				char task[30];
				sprintf(task,"-1 -1 -1 -1\n");
				write(bidding_to_host[k][1],task,strlen(task));
				wait(NULL);
			}
			break;
		}
	}
	int rank[player_num+1];
	for(int i=1;i<=player_num;i++)
		rank[i]=1;
	for(int i=1;i<=player_num;i++)
		for(int j=1;j<=player_num;j++)
		{
			if(i!=j)
			{
				if(grade[j]>grade[i])
					rank[i]++;
			}
		}

	for(int i=1;i<=player_num;i++)
		printf("%d %d \n",i,rank[i]);
	return 0;
}
