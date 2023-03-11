#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <iostream>

void PrintMatrixHalf(half* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", __half2float(arr[i * cols + j]));
		printf("\n");
	}
	printf("\n");
}

void PrintMatrixFloat(float* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

__global__ void curandNormalize(float* temp, half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(min + range * temp[index]);
}

void curandGenerateUniformEx(curandGenerator_t generator, half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	/*curandGenerateUniform(generator, output, size >> 1);
	curandNormalize <<<0.0009765625f * size + 1, 1024>>> (output, size, min, max - min);*/

	float* temp;
	cudaMalloc(&temp, size << 2);
	curandGenerateUniform(generator, temp, size);
	curandNormalize <<<0.0009765625f * size + 1, 1024>>> (temp, output, size, min, max - min);
	cudaFree(temp);
}