#include <stdio.h>
#include <vector>
#include <iostream>
#include <sys/types.h>    
#include <sys/stat.h>    
#include <fcntl.h>
using namespace std;
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
int main(int argc, char const *argv[])
{
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
			allstops.push_back(newstop);
	}
	printf("%d\n",int(allstops.size()));
	int tot=0,sbi=0,bemp=0;
	for(int i=0;i<allstops.size();i++)
	{
		tot+=allstops[i].tot;
		sbi+=allstops[i].sbi;
		bemp+=allstops[i].bemp;
	}
	printf("%d %d %d\n",tot,sbi,bemp );

	return 0;
}