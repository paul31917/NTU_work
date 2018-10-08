#include <stdio.h>

int main(int argc, char const *argv[])
{
	int g,x,p;
	while(scanf("%d%d%d",&g,&x,&p)!=EOF)
	{
		int ans=1;
		for(int i=1;i<=x;i++)
			ans=ans*g%p;
		printf("%d ^%d= %d (mod %d)\n",g,x,ans,p);
	}
	return 0;
}