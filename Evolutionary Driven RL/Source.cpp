#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <cuda_runtime.h>

int main()
{
	float* cpusrc = new float[1000000];
	float* cpudst = new float[1000000];
	float* gpusrc;
	cudaMalloc(&gpusrc, 1000000 * sizeof(float));
	float* gpudst;
	cudaMalloc(&gpudst, 1000000 * sizeof(float));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	
	for (uint32_t i = 1000; i--;)
	{
		cudaMemcpy(gpudst, cpusrc, 1000000 * sizeof(float), cudaMemcpyHostToDevice);
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken: %f ms\n", milliseconds);

	
	cudaEventRecord(start);
	
	for (uint32_t i = 1000; i--;)
	{
		memcpy(cpudst, cpusrc, 1000000 * sizeof(float));
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken: %f ms\n", milliseconds);

	return 0;
}
