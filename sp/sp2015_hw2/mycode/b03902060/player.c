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
int main(int argc, char const *argv[])
{
	int host_id=atoi(argv[1]);
	char player_index=argv[2][0];
	int random_key=atoi(argv[3]);
	char FIFOname1[20];
	sprintf(FIFOname1,"host%d_%c.FIFO",host_id,player_index);
	char FIFOname2[20];
	sprintf(FIFOname2,"host%d.FIFO",host_id);
	int fd1=open(FIFOname1,O_RDONLY);
	int fd2=open(FIFOname2,O_WRONLY);
	int money[4];
	char allmoney[100];
	for(int j=0;j<10;j++)
	{
		read(fd1, allmoney,100);
		sscanf(allmoney,"%d%d%d%d\n",&money[0],&money[1],&money[2],&money[3]);	
		int callmoney=0;
		char whotopay='A'; 
		for(int i=0;i<4;i++)
		{
			if(money[i]>callmoney)
			{
				callmoney=money[i];
				whotopay='A'+i;
			}
		}
		char returncommand[30];
		if(player_index==whotopay)
			sprintf(returncommand,"%c %d %d\n",player_index,random_key,callmoney);
		else
			sprintf(returncommand,"%c %d %d\n",player_index,random_key,0);
		write(fd2,returncommand,strlen(returncommand));
	}
	return 0;
	
}