#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
int numinthread;
int testdatanum;
typedef struct 
{
	int sort_num;
	int *pointer;
}Num_pointer;

pthread_mutex_t mutex;
int compare(const void *a,const void* b)
{
	int *num1=(int*)a;
	int *num2=(int*)b;
	if(*num1>(*num2))
		return 1;
	if(*num1<(*num2))
		return -1;
	return 0;
}
void* sort_first(void* n)
{	
	pthread_mutex_lock(&mutex);
	Num_pointer *np= (Num_pointer *)n; 
	fprintf(stdout, "Handling elements:\n");
	for(int i=0;i<(np->sort_num);i++)
		fprintf(stdout, "%d ",*(((np->pointer))+i));
	fprintf(stdout, "\nSorted %d elements.\n",np->sort_num);
	qsort(np->pointer,np->sort_num,sizeof(int),compare);
	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
}
void* sort_second(void* n)
{
	pthread_mutex_lock(&mutex);
	Num_pointer *np= (Num_pointer *)n; 
	fprintf(stdout, "Handling elements:\n");
	for(int i=0;i<(np->sort_num);i++)
		fprintf(stdout, "%d ",*((np->pointer)+i));
	fprintf(stdout, "\n" );
	int num1 = numinthread/2;
	int num2 = (np->sort_num)-num1;
	int *copy = (int*)( malloc(sizeof(int) * (np->sort_num)));
	for(int i=0;i<(np->sort_num);i++)
		*(copy+i)=*((np->pointer)+i);
	int *pointer1 = copy;
	int *pointer2 = copy+numinthread/2;
	int duplicate=0;
	int count1=0;
	int count2=0;
	for(int i=0;i<(np->sort_num);i++)
	{
		if(count1==num1)
		{
			*((np->pointer)+i)=*(pointer2+count2);
			count2++;
		}
		else if(count2==num2)
		{
			*((np->pointer)+i)=*(pointer1+count1);
			count1++;
		}
		else
		{
			if(*(pointer1+count1) == *(pointer2+count2))
			{
				duplicate++;
				*((np->pointer)+i) = (*(pointer1+count1));
				count1++;
			}
			else if(*(pointer1+count1) < *(pointer2+count2))
			{
				*((np->pointer)+i)=(*(pointer1+count1));
				count1++;
			}
			else if(*(pointer1+count1) > *(pointer2+count2))
			{
				*((np->pointer)+i)=(*(pointer2+count2));
				count2++;
			}
		}
	}
	fprintf(stdout, "Merged %d and %d elements with %d duplicates.\n",num1,num2,duplicate);
	pthread_mutex_unlock(&mutex);
	pthread_exit(NULL);
}
int main(int argc, char const *argv[])
{
	numinthread=atoi(argv[1]);
	scanf("%d",&testdatanum);
	int *testdata;
	testdata=(int*)malloc(sizeof(int)*testdatanum);
	for(int i=0;i<testdatanum;i++)
		scanf("%d",testdata+i);
	int threadnum;
	int flag=0;
	if(testdatanum % numinthread==0)
		threadnum = testdatanum / numinthread;
	else
	{
		threadnum = testdatanum / numinthread+1;
		flag=1;
	}
	pthread_t tid[threadnum];
	Num_pointer np[threadnum];
	pthread_mutex_init(&mutex,NULL);
	for(int i=0;i<threadnum;i++)
	{
		if(flag==1 && i==(threadnum-1))
		{
			np[i].sort_num=(testdatanum % numinthread);
			np[i].pointer=testdata+ i*numinthread;
		}
		else
		{
			np[i].sort_num=numinthread;
			np[i].pointer=testdata+i*numinthread;
		}
		pthread_create(&tid[i],NULL,sort_first,(void*)(&np[i]));
	}
	for(int i=0;i<threadnum;i++)
		pthread_join(tid[i],NULL);
	while(1)
	{
		numinthread*=2;
		flag=0;
		int newthread=threadnum/2;
		if(threadnum%2==0)
			threadnum/=2;
		else
		{
			threadnum/=2;
			threadnum++;
		}
		if(newthread==0)
			break;
		pthread_t tid2[newthread];
		if(testdatanum % numinthread !=0)
			flag=1;
		Num_pointer np2[threadnum];
		for(int i=0;i<newthread;i++)
		{
			if(flag==1 && i==(newthread-1))
			{
				np2[i].sort_num=(testdatanum % numinthread);
				np2[i].pointer=testdata+ i*numinthread;
			}
			else
			{
				np2[i].sort_num=numinthread;
				np2[i].pointer=testdata+i*numinthread;
			}
			pthread_create(&tid2[i],NULL,sort_second,(void*)(&np2[i]));
		}
		for(int i=0;i<newthread;i++)
			pthread_join(tid2[i],NULL);
	}
	for(int i=0;i<testdatanum;i++)
	{
		if(i==testdatanum-1)
			fprintf(stdout, "%d\n",*(testdata+i) );
		else
			fprintf(stdout, "%d ",*(testdata+i));
		
	}
	return 0;
}