#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>    
#include <sys/stat.h>    
#include <fcntl.h>
#include <time.h>
using namespace std;
typedef struct 
{
	char t_id[10];
	char student_id[30];
	char stop_nameen[100];
	char time[20];
	char date[20];
	int bike_count;
	//char now_date[20];
}Ticket;
typedef struct 
{
	char id[10];
	char company[10];
	char type[20];
	char situation[10];
}Bike;
typedef struct Stop
{
	char sno[10000];//站點代號
	char sna[10000];//場站名稱(中文)
	int tot;//場站總停車格
	int sbi;//場站目前車輛數量 
	char sarea[10000];//場站區域(中文)
	long long int mday;//資料更新時間
	double lat;//緯度
	double lng;//經度
	char ar[10000];//地(中文)
	char sareaen[10000];//場站區域(英文)
	char snaen[10000];//場站名稱(英文)
	char aren[10000];//地址(英文)
	int bemp;//空位數量
	int act;//全站禁用狀態
}Stop;
typedef struct 
{
	char student_id[20];
	char student_name[20];
	char sexual[10];
	char major_in[20];
}Student;
#define Maxbike 759
char* createdate()
{
	int month=(rand()%12+1);
	int day=(rand()%25+1);
	char date[20];
	sprintf(date,"2016-%02d-%02d",month,day);
	return date;
}

int main(int argc, char const *argv[])
{
	srand(time(NULL));
	FILE *fp;
	fp=fopen(argv[1],"r");
	char buffer[10000];
	vector<Stop> allstops;
	while(fscanf(fp,"%s",buffer)!=EOF)
	{
		Stop newstop;
		fscanf(fp,"%s ",buffer);
		fgets(newstop.sno,10000,fp);
		fscanf(fp,"%s ",buffer);
		fgets(newstop.sna,10000,fp);
		fscanf(fp,"%s %d",buffer,&newstop.tot);
		fscanf(fp,"%s %d",buffer,&newstop.sbi);
		fscanf(fp,"%s ",buffer);
		fgets(newstop.sarea,10000,fp);
		fscanf(fp,"%s %lld",buffer,&newstop.mday);
		fscanf(fp,"%s %lf",buffer,&newstop.lat);
		fscanf(fp,"%s %lf",buffer,&newstop.lng);
		fscanf(fp,"%s ",buffer);
		fgets(newstop.ar,10000,fp);
		fscanf(fp,"%s ",buffer);
		fgets(newstop.sareaen,10000,fp);
		//newstop.sareaen[10]='\n';
		//printf("%s\n",newstop.sareaen);
		fscanf(fp,"%s ",buffer);
		fgets(newstop.snaen,10000,fp);
		fscanf(fp,"%s ",buffer);
		fgets(newstop.aren,10000,fp);
		fscanf(fp,"%s %d",buffer,&newstop.bemp);
		fscanf(fp,"%s %d",buffer,&newstop.act);
		if(newstop.sareaen[0]=='D' && newstop.sareaen[1]=='a' && newstop.sareaen[2]=='a')
		{
			allstops.push_back(newstop);
			//printf("%s;%s;%d;%d;%d\n",newstop.snaen,newstop.aren,newstop.tot,newstop.sbi,newstop.bemp);
		}
	}
	fclose(fp);
	fp=fopen(argv[2],"r");
	vector<Bike> allbike;
	Bike newbike;
	while(fscanf(fp,"%s",buffer)!=EOF)
	{
		Bike newbike;
		fscanf(fp,"%s",newbike.id);
		fscanf(fp,"%s",buffer);
		fscanf(fp,"%s",newbike.company);
		fscanf(fp,"%s",buffer);
		fscanf(fp,"%s",newbike.type);
		fscanf(fp,"%s",buffer);
		fscanf(fp,"%s",newbike.situation);
		allbike.push_back(newbike);
	}
	fclose(fp);
	fp=fopen(argv[3],"r");
	vector<Student> allstudent;
	Student newstudent;
	while(fscanf(fp,"%s",newstudent.student_id)!=EOF)
	{
		fscanf(fp,"%s",newstudent.student_name);
		fscanf(fp,"%s",newstudent.sexual);
		fscanf(fp,"%s",newstudent.major_in);
		allstudent.push_back(newstudent);
	}
	fclose(fp);
	vector<Ticket> allticket;
	fp=fopen(argv[4],"r");
	for(int i=0;i<200;i++)
	{
		Ticket newticket;
		fscanf(fp,"%s",newticket.t_id);
		fscanf(fp,"%s",newticket.student_id);
		fscanf(fp,"%s",newticket.time);
		fscanf(fp,"%s",newticket.stop_nameen);
		fscanf(fp,"%s",newticket.date);
		fscanf(fp,"%d",&newticket.bike_count);
		allticket.push_back(newticket);
	}
	int record_id=1;
	for(int i=0;i<200;i++)
	{
		int a=rand()%30;
		int c=rand()%60;
		int d=rand()%(allstops.size());
		int e=rand()%30+30;
		int b=rand()%2;
		for(int j=0;j<allticket[i].bike_count;j++)
		{
			int g=rand()%60;
			printf("%03d;",record_id);
			printf("%03d;",i+1);
			printf("%s;",allbike[record_id].id);
			printf("%s ",allticket[i].date);
			printf("%c%c:%02d:%02d;",allticket[i].time[0],allticket[i].time[1],a,c);
			printf("%s;",allstops[d].snaen);
			printf("%s ",allticket[i].date);
			printf("%c%c:%02d:%02d\n",allticket[i].time[0],allticket[i].time[1],e,g);
			record_id++;
		}
	}
	

	return 0;
}