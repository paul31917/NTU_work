#include <stdio.h>
#include <math.h>
int main(int argc, char const *argv[])
{
	int g,p,h;
	g=atoi(argv[1]);
	h=atoi(argv[2]);
	p=atoi(argv[3]);
	int N=0;
	int num=1;
	for(N=1;N<=p-1;N++)
	{
		num = num * g % p;
		if(num==1)
			break;
	}
	printf("N=%d\n",N );
	int n=1+(int)sqrt(N);
	printf("%d\n",n);
	int n1=1;
	for(int m=N-n;m>0;m--)
	{
		n1 = n1 * g % p;
	}
	printf("n1=%d\n",n1);
	int babystep[n+1];
	babystep[0]=1;
	printf("List1: ");
	for(int i=0;i<n;i++)
	{
		printf("%5d ",babystep[i]);
		babystep[i+1]=(babystep[i]*g)%p;
	}
	printf("%5d \n",babystep[n]);
	int giantstep=h;
	int x=0;
	printf("List2: ");
	for(int m=0;m<n+1;m++)
	{
		printf("%5d ",giantstep);
		int k;
		for(k=0;k<n+1;k++)
		{
			if(giantstep==babystep[k])
				break;
		}
		if(k!=(n+1))
		{
			x=k+n*m;
			printf("\n");
			break;
		}
		giantstep = giantstep * n1 % p;
	}

	printf("x=%d\n",x);
	return 0;
}