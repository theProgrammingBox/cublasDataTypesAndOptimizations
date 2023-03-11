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
	curandSetPseudoRandomGeneratorSeed(curandGenerator, 1234ULL);

	const uint32_t INPUTS = 3;
	const uint32_t OUTPUTS = 2;

	float* gpuInputMatrix;
	float* gpuWeightMatrix;
	float* gpuOutputMatrix;

	float* cpuInputMatrix = new float[INPUTS];
	float* cpuWeightMatrix = new float[INPUTS * OUTPUTS];
	float* cpuOutputMatrix = new float[OUTPUTS];

	cudaMalloc(&gpuInputMatrix, INPUTS << 2);
	cudaMalloc(&gpuWeightMatrix, INPUTS * OUTPUTS << 2);
	cudaMalloc(&gpuOutputMatrix, OUTPUTS << 2);

	curandGenerateUniformEx(curandGenerator, gpuInputMatrix, INPUTS, -1.0f, 1.0f);
	curandGenerateUniformEx(curandGenerator, gpuWeightMatrix, INPUTS * OUTPUTS, -1.0f, 1.0f);

	float alpha = 1.0f;
	float beta = 0.0f;
	cublasGemmStridedBatchedEx(
		cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
		OUTPUTS, 1, INPUTS,
		&alpha,
		gpuWeightMatrix, CUDA_R_32F, OUTPUTS, 0,
		gpuInputMatrix, CUDA_R_32F, INPUTS, 0,
		&beta,
		gpuOutputMatrix, CUDA_R_32F, OUTPUTS, 0,
		1, CUDA_R_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP
	);
	
	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, INPUTS << 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, INPUTS * OUTPUTS << 2, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuOutputMatrix, gpuOutputMatrix, OUTPUTS << 2, cudaMemcpyDeviceToHost);
	
	PrintMatrix(cpuInputMatrix, INPUTS, 1, "Input");
	PrintMatrix(cpuWeightMatrix, INPUTS, OUTPUTS, "Weight");
	PrintMatrix(cpuOutputMatrix, OUTPUTS, 1, "Output");
	
	curandDestroyGenerator(curandGenerator);
	cublasDestroy(cublasHandle);

	return 0;
}