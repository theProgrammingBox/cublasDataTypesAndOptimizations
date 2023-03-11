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

	const uint32_t INPUTS = 3;
	const uint32_t OUTPUTS = 7;

	//
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

	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, INPUTS << 1, cudaMemcpyDeviceToHost);
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
	//

	cublasDestroy(cublasHandle);
	curandDestroyGenerator(curandGenerator);

	return 0;
}