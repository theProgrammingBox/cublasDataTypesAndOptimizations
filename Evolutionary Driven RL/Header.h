﻿#pragma once
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cublas_v2.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

void PrintMatrixf32(float* arr, uint32_t rows, uint32_t cols, const char* label)
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

__global__ void CurandNormalizef32(float* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = float(*(uint32_t*)(output + index) * range + min);
}

void CurandGenerateUniformf32(curandGenerator_t generator, float* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, size);
	CurandNormalizef32 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 2.3283064365387e-10f);
}

__global__ void GpuReluf32(float* input, float* output, uint32_t size)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size && *(uint32_t*)(input + index) >> 31)
	{
		output[index] = 0;
	}
}

void Reluf32(float* input, float* output, uint32_t size)
{
	cudaMemcpy(output, input, size << 2, cudaMemcpyDeviceToDevice);
	GpuReluf32 << <std::ceil(0.0009765625f * size), 1024 >> > (input, output, size);
}

////////////////////////////////////////////////////////////////

void PrintMatrixf16(__half* arr, uint32_t rows, uint32_t cols, const char* label)
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

__global__ void CurandNormalizef16(__half* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __float2half(*(uint16_t*)(output + index) * range + min);
}

void CurandGenerateUniformf16(curandGenerator_t generator, __half* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 1) + (size & 1));
	CurandNormalizef16 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 0.0000152590218967f);
}

__global__ void GpuReluf16(__half* input, __half* output, uint32_t size)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size && *(uint16_t*)(input + index) >> 15)
	{
		output[index] = 0;
	}
}

void Reluf16(__half* input, __half* output, uint32_t size)
{
	cudaMemcpy(output, input, size << 1, cudaMemcpyDeviceToDevice);
	GpuReluf16 << <std::ceil(0.0009765625f * size), 1024 >> > (input, output, size);
}

////////////////////////////////////////////////////////////////

void PrintMatrixf8(__nv_fp8_e4m3* arr, uint32_t rows, uint32_t cols, const char* label)
{
	printf("%s:\n", label);
	for (uint32_t i = 0; i < rows; i++)
	{
		for (uint32_t j = 0; j < cols; j++)
			printf("%8.3f ", (float)arr[i * cols + j]);
		printf("\n");
	}
	printf("\n");
}

__global__ void CurandNormalizef8(__nv_fp8_e4m3* output, uint32_t size, float min, float range)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size)
		output[index] = __nv_fp8_e4m3(*(uint8_t*)(output + index) * range + min);
}

void CurandGenerateUniformf8(curandGenerator_t generator, __nv_fp8_e4m3* output, uint32_t size, float min = -1.0f, float max = 1.0f)
{
	curandGenerate(generator, (uint32_t*)output, (size >> 2) + bool(size & 3));
	CurandNormalizef8 << <std::ceil(0.0009765625f * size), 1024 >> > (output, size, min, (max - min) * 0.00392156862745f);
}

__global__ void GpuReluf8(__nv_fp8_e4m3* input, __nv_fp8_e4m3* output, uint32_t size)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if (index < size && *(uint8_t*)(input + index) >> 7)
	{
		output[index] = (__nv_fp8_e4m3)0;
	}
}

void Reluf8(__nv_fp8_e4m3* input, __nv_fp8_e4m3* output, uint32_t size)
{
	cudaMemcpy(output, input, size, cudaMemcpyDeviceToDevice);
	GpuReluf8 << <std::ceil(0.0009765625f * size), 1024 >> > (input, output, size);
}