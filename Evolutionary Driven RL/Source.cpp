#include "Header.cuh"

/*
Time taken for cpu to cpu: 174.233307 ms
Time taken for cpu to gpu: 670.819031 ms
Time taken for gpu to cpu: 967.830139 ms
Time taken for gpu to gpu: 37.351521 ms
*/

void F16Default(cublasHandle_t cublasHandle, curandGenerator_t curandGenerator, const uint32_t INPUTS = 3, const uint32_t OUTPUTS = 7)
{
	__half* gpuInputMatrix;
	__half* gpuWeightMatrix;
	__half* gpuProductMatrix;
	__half* gpuReluMatrix;

	cudaMalloc(&gpuInputMatrix, INPUTS << 1);
	cudaMalloc(&gpuWeightMatrix, INPUTS * OUTPUTS << 1);
	cudaMalloc(&gpuProductMatrix, OUTPUTS << 1);
	cudaMalloc(&gpuReluMatrix, OUTPUTS << 1);

	__half* cpuInputMatrix = (__half*)malloc(INPUTS << 1);
	__half* cpuWeightMatrix = (__half*)malloc(INPUTS * OUTPUTS << 1);
	__half* cpuOutputMatrix = (__half*)malloc(OUTPUTS << 1);
	__half* cpuReluMatrix = (__half*)malloc(OUTPUTS << 1);

	CurandGenerateUniformf16(curandGenerator, gpuInputMatrix, INPUTS);
	CurandGenerateUniformf16(curandGenerator, gpuWeightMatrix, INPUTS * OUTPUTS);

	const __half alpha = 1.0f;
	const __half beta = 0.0f;

	cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		OUTPUTS, 1, INPUTS,
		&alpha,
		gpuWeightMatrix, CUDA_R_16F, OUTPUTS, 0,
		gpuInputMatrix, CUDA_R_16F, INPUTS, 0,
		&beta,
		gpuProductMatrix, CUDA_R_16F, OUTPUTS, 0,
		1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
	);

	Reluf16(gpuProductMatrix, gpuReluMatrix, OUTPUTS);

	/**/cudaMemcpy(cpuInputMatrix, gpuInputMatrix, INPUTS << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, INPUTS * OUTPUTS << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuProductMatrix, OUTPUTS << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuReluMatrix, gpuReluMatrix, OUTPUTS << 1, cudaMemcpyDeviceToHost);

	PrintMatrixf16(cpuInputMatrix, 1, INPUTS, "Input");
	PrintMatrixf16(cpuWeightMatrix, INPUTS, OUTPUTS, "Weight");
	PrintMatrixf16(cpuOutputMatrix, 1, OUTPUTS, "Output");
	PrintMatrixf16(cpuReluMatrix, 1, OUTPUTS, "Relu");

	cudaFree(gpuInputMatrix);
	cudaFree(gpuWeightMatrix);
	cudaFree(gpuProductMatrix);

	free(cpuInputMatrix);
	free(cpuWeightMatrix);
	free(cpuOutputMatrix);
}

void F8Default(cublasHandle_t cublasHandle, curandGenerator_t curandGenerator, const uint32_t INPUTS = 3, const uint32_t OUTPUTS = 7)
{
	__nv_fp8_e4m3* gpuInputMatrix;
	__nv_fp8_e4m3* gpuWeightMatrix;
	__nv_fp8_e4m3* gpuProductMatrix;
	__nv_fp8_e4m3* gpuReluMatrix;
	
	cudaMalloc(&gpuInputMatrix, INPUTS);
	cudaMalloc(&gpuWeightMatrix, INPUTS * OUTPUTS);
	cudaMalloc(&gpuProductMatrix, OUTPUTS);
	cudaMalloc(&gpuReluMatrix, OUTPUTS);
	
	__nv_fp8_e4m3* cpuInputMatrix = (__nv_fp8_e4m3*)malloc(INPUTS);
	__nv_fp8_e4m3* cpuWeightMatrix = (__nv_fp8_e4m3*)malloc(INPUTS * OUTPUTS);
	__nv_fp8_e4m3* cpuOutputMatrix = (__nv_fp8_e4m3*)malloc(OUTPUTS);
	__nv_fp8_e4m3* cpuReluMatrix = (__nv_fp8_e4m3*)malloc(OUTPUTS);
	
	CurandGenerateUniformf8(curandGenerator, gpuInputMatrix, INPUTS);
	CurandGenerateUniformf8(curandGenerator, gpuWeightMatrix, INPUTS * OUTPUTS);

	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, INPUTS, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, INPUTS * OUTPUTS, cudaMemcpyDeviceToHost);
	
	PrintMatrixf8(cpuInputMatrix, 1, INPUTS, "Input");
	PrintMatrixf8(cpuWeightMatrix, INPUTS, OUTPUTS, "Weight");
}

int main()
{
	cublasHandle_t cublasHandle;
	cublasCreate(&cublasHandle);

	curandGenerator_t curandGenerator;
	curandCreateGenerator(&curandGenerator, CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(curandGenerator, std::chrono::high_resolution_clock::now().time_since_epoch().count());

	const uint32_t INPUTS = 3;
	const uint32_t OUTPUTS = 7;
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	float milliseconds;
	
	cudaEventRecord(start);
	for (uint32_t itr = 1; itr--;)
		F16Default(cublasHandle, curandGenerator, INPUTS, OUTPUTS);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Time taken for F16Default: %f ms\n", milliseconds);

	F8Default(cublasHandle, curandGenerator, INPUTS, OUTPUTS);

	cublasDestroy(cublasHandle);
	curandDestroyGenerator(curandGenerator);

	return 0;
}