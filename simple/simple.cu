#include <iostream>
#include <math.h>
#include <stdio.h>

/* Notes - I encountered some weird and unexplainable issues using this
 * 6-22-2025 under Fedora 42
 * I got segfaults and core dumped due to zero values getting
 *  returned by cudaMallocManaged() -- even a trivial example like
 *  this should check return values.  I fooled around reinstalling
 *  the nvidia-open package.  Then I tried this again.  I ran it two
 *  times and got zero values, then it started to work!!
 * I see chatter like this in the system log file:
 *  NVRM: nvCheckOkFailedNoLog: Check failed: Out of memory
 */

#ifdef notdef
// Kernel function to add the elements of two arrays
// on the device
// Runs in 103,733,432
__global__
void add(int n, float *x, float *y)
{
 for (int i = 0; i < n; i++)
   y[i] = x[i] + y[i];
}
#endif

// with 1,   1 runs in 71,312,495
// with 1, 256 runs in 5,010,811
// with 1, 1024 runs   3,990,971

__global__
void gpu_addX (int n, float *x, float *y)
{
  int index = threadIdx.x;
  int stride = blockDim.x;

  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}

// without prefetch x, 256 runs 3,457,884
// with prefetch x, 256    runs    39,296
__global__
void gpu_add (int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    y[i] = x[i] + y[i];
}

// The same function, but it runs on the CPU
void cpu_add(int n, float *x, float *y)
{
 for (int i = 0; i < n; i++)
   y[i] = x[i] + y[i];
}

void cpu_demo ( int N )
{
 float *x = new float[N];
 float *y = new float[N];

 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }

 // Run kernel on 1M elements on the CPU
 cpu_add(N, x, y);

 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 std::cout << "Max error: " << maxError << std::endl;

 // Free memory
 delete [] x;
 delete [] y;
}

void gpu_demo ( int N )
{
 float *x, *y;

 // Allocate Unified Memory – accessible from CPU or GPU
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));

 if ( x == NULL || y == NULL ) {
 	printf ( "Cannot allocate memory\n" );
	exit ( 1 );
 }

 printf ( "x: %08x\n", x );
 printf ( "y: %08x\n", y );

 printf ( "Load the arrays\n" );
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }

// Run kernel on 1M elements on the GPU
// add<<<1, 1>>>(N, x, y);
// add<<<1, 256>>>(N, x, y);
// gpu_add<<<1, 1024>>>(N, x, y);

// Prefetch the x and y arrays to the GPU
 cudaMemPrefetchAsync(x, N*sizeof(float), 0, 0);
 cudaMemPrefetchAsync(y, N*sizeof(float), 0, 0);

 int blockSize = 256;
 int numBlocks = (N + blockSize - 1) / blockSize;
 gpu_add<<<numBlocks, blockSize>>>(N, x, y);

 // Wait for GPU to finish before accessing on host
 cudaDeviceSynchronize();
 printf ( "Add is done\n" );

 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++) {
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 }
 std::cout << "Max error: " << maxError << std::endl;

 // Free memory
 cudaFree(x);
 cudaFree(y);
}

int main(void)
{
  int N = 1<<20;

  // cpu_demo ( N );
  gpu_demo ( N );

  return 0;
}

/* THE END */
