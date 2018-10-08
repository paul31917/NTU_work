#include <stdio.h>
#include <math.h>
#include <stdlib.h>
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}
int expmod(long long int a,int e,int n)
{
	int result=1;
	for(int i=1;i<=e;i++)
		result=result*a%n;
	return result;
}
int main(int argc, char const *argv[])
{
  // 計算2^b! - 1，存入a中。
    
    for(int i=3;i<30;i+=2)
    {
    	if(i%3==0)
    		continue;
    	int b=3;
    	int p=1;
    	int n = pow(2,i)-1;
    	while(p == 1 || p == n)
    	{
    		if(b>=sqrt(n))
   			{
   				printf("2^%d-1=%d is a prime !\n",i,n);
   				break;
   			}
      		//printf("b=%d\n",b );
    		long long int a = 2;
    		for (int e=2; e <= b; e++)
    	   		a = expmod(a, e, n);    // a = a^e % n
     		//printf("a=%lld\n",a);
    		p = gcd(a-1, n);    // gcd(2^b! - 1, n)
    		if(!(p == 1 || p == n))
    			printf("p=%d b=%d \n",p,b);
   			b++;
    	}    
    }
	return 0;
}