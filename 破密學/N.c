#include <stdio.h>
#include <stdlib.h>
#include <math.h>
int gcd(int a, int b) {
    return b == 0 ? a : gcd(b, a % b);
}
int main(int argc, char const *argv[])
{
	int N=atoi(argv[1]);
	int a=atoi(argv[2]);
	int b=atoi(argv[3]);
	printf("%d %d\n",gcd(N,a+b),gcd(N,a-b));
		
		
	
	return 0;
}