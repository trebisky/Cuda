#include <stdio.h>

__global__ void cuda_hello(){
	int i;

	for ( i=0 ; i<4; i++ )
		printf("Hello World %d from GPU!\n", i+1 );
}

int main() {
    cuda_hello<<<1,1>>>(); 
	cudaDeviceSynchronize();
	printf ( "Goodbye\n" );
    return 0;
}
