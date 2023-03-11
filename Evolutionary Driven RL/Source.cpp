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

	/*float in = 1.0f / 3.0f;
	half cpuInput = __float2half(in);
	printf("Input: %f\n", in);
	printf("Input: %f\n", __half2float(cpuInput));*/

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
	
	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, INPUTS << 1, cudaMemcpyDeviceToHost);
	cudaMemcpy(cpuWeightMatrix, gpuWeightMatrix, INPUTS * OUTPUTS << 1, cudaMemcpyDeviceToHost);
	PrintMatrixHalf(cpuInputMatrix, INPUTS, 1, "Input");
	PrintMatrixHalf(cpuWeightMatrix, INPUTS, OUTPUTS, "Weight");

	/*float alpha = 1.0f;
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
	PrintMatrix(cpuOutputMatrix, OUTPUTS, 1, "Output");*/
	
	curandDestroyGenerator(curandGenerator);
	cublasDestroy(cublasHandle);

	return 0;
}