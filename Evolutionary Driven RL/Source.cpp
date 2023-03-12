#include "Header.cuh"

/*
Time taken for cpu to cpu: 174.233307 ms
Time taken for cpu to gpu: 670.819031 ms
Time taken for gpu to cpu: 967.830139 ms
Time taken for gpu to gpu: 37.351521 ms
*/

int main()
{
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, std::chrono::high_resolution_clock::now().time_since_epoch().count());

	const uint32_t INPUTS = 17;
	const uint32_t OUTPUTS = 17;

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;

	__nv_fp8_e4m3* gpuInputMatrixf8;
	__nv_fp8_e4m3* gpuWeightMatrixf8;
	__nv_fp8_e4m3* gpuProductMatrixf8;
	__nv_fp8_e4m3* gpuReluMatrixf8;

	cudaMalloc(&gpuInputMatrixf8, INPUTS);
	cudaMalloc(&gpuWeightMatrixf8, INPUTS * OUTPUTS);
	cudaMalloc(&gpuProductMatrixf8, OUTPUTS);
	cudaMalloc(&gpuReluMatrixf8, OUTPUTS);

	__nv_fp8_e4m3* cpuInputMatrixf8 = (__nv_fp8_e4m3*)malloc(INPUTS);
	__nv_fp8_e4m3* cpuWeightMatrixf8 = (__nv_fp8_e4m3*)malloc(INPUTS * OUTPUTS);
	__nv_fp8_e4m3* cpuOutputMatrixf8 = (__nv_fp8_e4m3*)malloc(OUTPUTS);
	__nv_fp8_e4m3* cpuReluMatrixf8 = (__nv_fp8_e4m3*)malloc(OUTPUTS);

	cudaEventRecord(start);
	for (uint32_t itr = 100000; itr--;)
	{

		CurandGenerateUniformf8(curandGenerator, gpuInputMatrixf8, INPUTS);
		CurandGenerateUniformf8(curandGenerator, gpuWeightMatrixf8, INPUTS * OUTPUTS);

		const __nv_fp8_e4m3 alpha = (__nv_fp8_e4m3)1;
		const __nv_fp8_e4m3 beta = (__nv_fp8_e4m3)0;

		/*const float alpha = 1;
		const float beta = 0;*/

		cublasGemmStridedBatchedEx
		(
			cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			OUTPUTS, 1, INPUTS,
			&alpha,
			gpuWeightMatrixf8, CUDA_R_8F_E4M3, OUTPUTS, 0,
			gpuInputMatrixf8, CUDA_R_8F_E4M3, INPUTS, 0,
			&beta,
			gpuProductMatrixf8, CUDA_R_8F_E4M3, OUTPUTS, 0,
			1, CUDA_R_8F_E4M3, CUBLAS_GEMM_DEFAULT_TENSOR_OP
		);

		Reluf8(gpuProductMatrixf8, gpuReluMatrixf8, OUTPUTS);

		/*cudaMemcpy(cpuInputMatrixf8, gpuInputMatrixf8, INPUTS, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuWeightMatrixf8, gpuWeightMatrixf8, INPUTS * OUTPUTS, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuOutputMatrixf8, gpuProductMatrixf8, OUTPUTS, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuReluMatrixf8, gpuReluMatrixf8, OUTPUTS, cudaMemcpyDeviceToHost);

		PrintMatrixf8(cpuInputMatrixf8, 1, INPUTS, "Input");
		PrintMatrixf8(cpuWeightMatrixf8, INPUTS, OUTPUTS, "Weight");
		PrintMatrixf8(cpuOutputMatrixf8, 1, OUTPUTS, "Output");
		PrintMatrixf8(cpuReluMatrixf8, 1, OUTPUTS, "Relu");*/
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for F8Default: %f ms\n", milliseconds);

	cudaFree(gpuInputMatrixf8);
	cudaFree(gpuWeightMatrixf8);
	cudaFree(gpuProductMatrixf8);
	cudaFree(gpuReluMatrixf8);

	free(cpuInputMatrixf8);
	free(cpuWeightMatrixf8);
	free(cpuOutputMatrixf8);
	free(cpuReluMatrixf8);

	__half* gpuInputMatrixf16;
	__half* gpuWeightMatrixf16;
	__half* gpuProductMatrixf16;
	__half* gpuReluMatrixf16;

	cudaMalloc(&gpuInputMatrixf16, INPUTS << 1);
	cudaMalloc(&gpuWeightMatrixf16, INPUTS * OUTPUTS << 1);
	cudaMalloc(&gpuProductMatrixf16, OUTPUTS << 1);
	cudaMalloc(&gpuReluMatrixf16, OUTPUTS << 1);

	__half* cpuInputMatrixf16 = (__half*)malloc(INPUTS << 1);
	__half* cpuWeightMatrixf16 = (__half*)malloc(INPUTS * OUTPUTS << 1);
	__half* cpuOutputMatrixf16 = (__half*)malloc(OUTPUTS << 1);
	__half* cpuReluMatrixf16 = (__half*)malloc(OUTPUTS << 1);

	cudaEventRecord(start);
	for (uint32_t itr = 100000; itr--;)
	{

		CurandGenerateUniformf16(curandGenerator, gpuInputMatrixf16, INPUTS);
		CurandGenerateUniformf16(curandGenerator, gpuWeightMatrixf16, INPUTS * OUTPUTS);

		const __half alpha = 1;
		const __half beta = 0;

		/*const float alpha = 1;
		const float beta = 0;*/

		cublasGemmStridedBatchedEx
		(
			cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			OUTPUTS, 1, INPUTS,
			&alpha,
			gpuWeightMatrixf16, CUDA_R_16F, OUTPUTS, 0,
			gpuInputMatrixf16, CUDA_R_16F, INPUTS, 0,
			&beta,
			gpuProductMatrixf16, CUDA_R_16F, OUTPUTS, 0,
			1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
		);

		Reluf16(gpuProductMatrixf16, gpuReluMatrixf16, OUTPUTS);

		/*cudaMemcpy(cpuInputMatrixf16, gpuInputMatrixf16, INPUTS << 1, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuWeightMatrixf16, gpuWeightMatrixf16, INPUTS * OUTPUTS << 1, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuOutputMatrixf16, gpuProductMatrixf16, OUTPUTS << 1, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuReluMatrixf16, gpuReluMatrixf16, OUTPUTS << 1, cudaMemcpyDeviceToHost);

		PrintMatrixf16(cpuInputMatrixf16, 1, INPUTS, "Input");
		PrintMatrixf16(cpuWeightMatrixf16, INPUTS, OUTPUTS, "Weight");
		PrintMatrixf16(cpuOutputMatrixf16, 1, OUTPUTS, "Output");
		PrintMatrixf16(cpuReluMatrixf16, 1, OUTPUTS, "Relu");*/
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for F16Default: %f ms\n", milliseconds);

	cudaFree(gpuInputMatrixf16);
	cudaFree(gpuWeightMatrixf16);
	cudaFree(gpuProductMatrixf16);
	cudaFree(gpuReluMatrixf16);

	free(cpuInputMatrixf16);
	free(cpuWeightMatrixf16);
	free(cpuOutputMatrixf16);
	free(cpuReluMatrixf16);

	float* gpuInputMatrixf32;
	float* gpuWeightMatrixf32;
	float* gpuProductMatrixf32;
	float* gpuReluMatrixf32;

	cudaMalloc(&gpuInputMatrixf32, INPUTS << 2);
	cudaMalloc(&gpuWeightMatrixf32, INPUTS * OUTPUTS << 2);
	cudaMalloc(&gpuProductMatrixf32, OUTPUTS << 2);
	cudaMalloc(&gpuReluMatrixf32, OUTPUTS << 2);

	float* cpuInputMatrixf32 = (float*)malloc(INPUTS << 2);
	float* cpuWeightMatrixf32 = (float*)malloc(INPUTS * OUTPUTS << 2);
	float* cpuOutputMatrixf32 = (float*)malloc(OUTPUTS << 2);
	float* cpuReluMatrixf32 = (float*)malloc(OUTPUTS << 2);

	cudaEventRecord(start);
	for (uint32_t itr = 100000; itr--;)
	{
		CurandGenerateUniformf32(curandGenerator, gpuInputMatrixf32, INPUTS);
		CurandGenerateUniformf32(curandGenerator, gpuWeightMatrixf32, INPUTS * OUTPUTS);

		const float alpha = 1;
		const float beta = 0;

		cublasGemmStridedBatchedEx
		(
			cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
			OUTPUTS, 1, INPUTS,
			&alpha,
			gpuWeightMatrixf32, CUDA_R_32F, OUTPUTS, 0,
			gpuInputMatrixf32, CUDA_R_32F, INPUTS, 0,
			&beta,
			gpuProductMatrixf32, CUDA_R_32F, OUTPUTS, 0,
			1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
		);

		Reluf32(gpuProductMatrixf32, gpuReluMatrixf32, OUTPUTS);

		/*cudaMemcpy(cpuInputMatrixf32, gpuInputMatrixf32, INPUTS << 2, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuWeightMatrixf32, gpuWeightMatrixf32, INPUTS * OUTPUTS << 2, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuOutputMatrixf32, gpuProductMatrixf32, OUTPUTS << 2, cudaMemcpyDeviceToHost);
		cudaMemcpy(cpuReluMatrixf32, gpuReluMatrixf32, OUTPUTS << 2, cudaMemcpyDeviceToHost);

		PrintMatrixf32(cpuInputMatrixf32, 1, INPUTS, "Input");
		PrintMatrixf32(cpuWeightMatrixf32, INPUTS, OUTPUTS, "Weight");
		PrintMatrixf32(cpuOutputMatrixf32, 1, OUTPUTS, "Output");
		PrintMatrixf32(cpuReluMatrixf32, 1, OUTPUTS, "Relu");*/
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for F32Default: %f ms\n", milliseconds);

	cudaFree(gpuInputMatrixf32);
	cudaFree(gpuWeightMatrixf32);
	cudaFree(gpuProductMatrixf32);
	cudaFree(gpuReluMatrixf32);

	free(cpuInputMatrixf32);
	free(cpuWeightMatrixf32);
	free(cpuOutputMatrixf32);
	free(cpuReluMatrixf32);

	cublasDestroy(cublasHandle);
	curandDestroyGenerator(curandGenerator);

	return 0;
}