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

	const uint32_t rows = 3;
	const uint32_t cols = 4;

	float* gpuInputMatrix;
	float* gpuWeightMatrix;
	float* gpuOutputMatrix;

	float* cpuInputMatrix = new float[rows * cols];

	cudaMalloc(&gpuInputMatrix, rows * cols * sizeof(float));
	cudaMalloc(&gpuWeightMatrix, rows * cols * sizeof(float));
	cudaMalloc(&gpuOutputMatrix, rows * cols * sizeof(float));

	curandGenerateUniformEx(curandGenerator, gpuInputMatrix, rows * cols, -1, 1);
	cudaMemcpy(cpuInputMatrix, gpuInputMatrix, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);
	PrintMatrix(cpuInputMatrix, rows, cols, "Input Matrix");
	
	curandDestroyGenerator(curandGenerator);
	cublasDestroy(cublasHandle);

	return 0;
}