#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

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

__global__ void CurandNormalize(half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(*(uint16_t*)(output + index) * range + min);
}

void CurandGenerateUniformEx(curandGenerator_t generator, half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, size << 1);
	CurandNormalize <<<std::ceil(0.0009765625f * size), 1024>>> (output, size, min, (max - min) * 0.0000152590218967f);
}

__global__ void GpuRelu(half* input, half* output, uint32_t size)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size && *(uint16_t*)(input + index) >> 15)
	{
		output[index] = 0.0f;
	}
}

void Relu(half* input, half* output, uint32_t size)
{
	cudaMemcpy(output, input, size << 1, cudaMemcpyDeviceToDevice);
	GpuRelu << <std::ceil(0.0009765625f * size), 1024 >> > (input, output, size);
}

__global__ void GpuRelu2(half* input, half* output, uint32_t size)
{
	if (index < size)
	{
		uint16_t a = *(uint16_t*)(input + index);
		uint16_t b = (a >> 15) * a;
		memcpy(output + index, &b, 2);
	}
}

void Relu2(half* input, half* output, uint32_t size)
{
	GpuRelu2 <<<std::ceil(0.0009765625f * size), 1024>>> (input, output, size);
}