#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <string.h>
#include <time.h>
using namespace std;
typedef struct 
{
	char id[10];
	char company[10];
	char type[20];
	char situation[10];
}Bike;
#define Maxbike 759
int main(int argc, char const *argv[])
{
	srand(time(NULL));
	int A[Maxbike] ;
	for(int i=0;i<Maxbike;i++)
		A[i]=i+1;
	random_shuffle(A,A+Maxbike);

	vector<Bike> allbike;
	for(int i=0;i<Maxbike;i++)
	{
		Bike newbike;
		int a=(rand()%3);
		sprintf(newbike.id, "%03d",A[i]);
		if(a==0)
			strcpy(newbike.company, "Giant");
		else if(a==1)
			strcpy(newbike.company, "MeRiDa");
		else
			strcpy(newbike.company, "911");
		a=(rand()%2);
		if(a==0)
			strcpy(newbike.type, "Normal_Youbike");
		else 
			strcpy(newbike.type, "iNeed_FunCity");
		if(i<28)
			strcpy(newbike.situation,"bad");
		else
			strcpy(newbike.situation,"well");
		allbike.push_back(newbike);
	}
	for(int i=0;i<allbike.size();i++)
	{
		printf("id: %s\n",allbike[i].id);
		printf("company: %s\n",allbike[i].company);
		printf("type: %s\n",allbike[i].type);
		printf("situation: %s\n",allbike[i].situation);
		printf("\n");
	}
	
	return 0;
}