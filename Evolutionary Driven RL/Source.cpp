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
	const uint32_t OUTPUTS = 2;

	half* gpuInputMatrix;
	half* gpuWeightMatrix;
	half* gpuOutputMatrix;
	
	half* cpuInputMatrix = (half*)malloc(INPUTS << 1);
	half* cpuWeightMatrix = (half*)malloc(INPUTS * OUTPUTS << 1);
	half* cpuOutputMatrix = (half*)malloc(OUTPUTS << 1);

	cudaMalloc(&gpuInputMatrix, INPUTS << 1);
	cudaMalloc(&gpuWeightMatrix, INPUTS * OUTPUTS << 1);
	cudaMalloc(&gpuOutputMatrix, OUTPUTS << 1);
	
	curandGenerateUniformEx(curandGenerator, gpuInputMatrix, INPUTS);
	curandGenerateUniformEx(curandGenerator, gpuWeightMatrix, INPUTS * OUTPUTS);

	const half alpha = 1.0f;
	const half beta = 0.0f;
	
	cublasGemmStridedBatchedEx
	(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		OUTPUTS, 1, INPUTS,
		&alpha,
		gpuWeightMatrix, CUDA_R_16F, OUTPUTS, 0,
		gpuInputMatrix, CUDA_R_16F, INPUTS, 0,
		&beta,
		gpuOutputMatrix, CUDA_R_16F, OUTPUTS, 0,
		1, CUDA_R_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
	);

	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, INPUTS << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, INPUTS * OUTPUTS << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, OUTPUTS << 1, cudaMemcpyDeviceToHost);
	
	PrintMatrixHalf(cpuInputMatrix, 1, INPUTS, "Input");
	PrintMatrixHalf(cpuWeightMatrix, INPUTS, OUTPUTS, "Weight");
	PrintMatrixHalf(cpuOutputMatrix, 1, OUTPUTS, "Output");
	
	curandDestroyGenerator(curandGenerator);
	cublasDestroy(cublasHandle);
	
	cudaFree(gpuInputMatrix);
	cudaFree(gpuWeightMatrix);
	cudaFree(gpuOutputMatrix);
	
	free(cpuInputMatrix);
	free(cpuWeightMatrix);
	free(cpuOutputMatrix);

	return 0;
}