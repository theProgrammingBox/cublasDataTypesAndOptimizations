#pragma once
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <iostream>

void PrintMatrix(float* arr, uint32_t rows, uint32_t cols, const char* label) {
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

__global__ void curandNormalize(float* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = output[index] * range + min;
}

void curandGenerateUniformEx(curandGenerator_t generator, float* output, uint32_t size, float min, float max)
{
	curandGenerateUniform(generator, output, size);
	curandNormalize <<<0.0009765625f * size + 1, 1024>>> (output, size, min, max - min);
}