#include <cublas_v2.h>
#include <curand.h>
#include <iostream>
#include <cuda_runtime.h>

/*
Time taken for cpu to cpu: 174.233307 ms
Time taken for cpu to gpu: 670.819031 ms
Time taken for gpu to cpu: 967.830139 ms
Time taken for gpu to gpu: 37.351521 ms
*/

int main()
{
	cudaSetDevice(0);
	cublasHandle_t handle;
	cublasCreate(&handle);
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop, 0);
	size_t total_mem = prop.totalGlobalMem;
	printf("Total memory: %zu\n", total_mem);
	
	
	float* cpusrc = new float[1000000];
	float* cpudst = new float[1000000];
	float* gpusrc;
	cudaMalloc(&gpusrc, 1000000 * sizeof(float));
	float* gpudst;
	cudaMalloc(&gpudst, 1000000 * sizeof(float));
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;
	

	cudaEventRecord(start);

	for (uint32_t i = 1000; i--;)
	{
		memcpy(cpudst, cpusrc, 1000000 * sizeof(float));
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for cpu to cpu: %f ms\n", milliseconds);

	
	cudaEventRecord(start);
	
	for (uint32_t i = 1000; i--;)
	{
		cudaMemcpy(gpudst, cpusrc, 1000000 * sizeof(float), cudaMemcpyHostToDevice);
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for cpu to gpu: %f ms\n", milliseconds);

	
	cudaEventRecord(start);
	
	for (uint32_t i = 1000; i--;)
	{
		cudaMemcpy(cpusrc, gpudst, 1000000 * sizeof(float), cudaMemcpyDeviceToHost);
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for gpu to cpu: %f ms\n", milliseconds);

	
	cudaEventRecord(start);
	
	for (uint32_t i = 1000; i--;)
	{
		cudaMemcpy(gpudst, gpusrc, 1000000 * sizeof(float), cudaMemcpyDeviceToDevice);
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for gpu to gpu: %f ms\n", milliseconds);

	
	// now using cublasSetMatrix
	cudaEventRecord(start);

	for (uint32_t i = 1000; i--;)
	{
		cublasSetMatrix(1000, 1000, sizeof(float), cpusrc, 1000, gpusrc, 1000);
	}

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for cpu to gpu: %f ms\n", milliseconds);

	
	cublasDestroy(handle);
	cudaFree(gpusrc);
	cudaFree(gpudst);
	delete[] cpusrc;
	delete[] cpudst;

	return 0;
}
