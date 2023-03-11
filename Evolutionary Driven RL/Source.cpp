#include "Header.h"

int main()
{
	auto start = std::chrono::high_resolution_clock::now();
	uint32_t a = 0, b = 1, c = 0;
	for (uint32_t i = 1000000; i--;)
	{
		c = a + b;
		a = b;
		b = c;
	}
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

	
	float* input = new float[10];
	float* output = new float[10];


	start = std::chrono::high_resolution_clock::now();
	for (uint32_t i = 1000000; i--;)
	{
		cpuGenerateUniform(input, 10, -1, 1);
		cpuRelu(input, output, 10);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	printf("Time taken by function: %llu microseconds\n", duration.count());


	start = std::chrono::high_resolution_clock::now();
	for (uint32_t i = 1000000; i--;)
	{
		cpuGenerateUniform(input, 10, -1, 1);
		cpuRelu2(input, output, 10);
	}
	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
	printf("Time taken by function: %llu microseconds\n", duration.count());

	

	cpuGenerateUniform(input, 10, -2, 2);
	cpuRelu(input, output, 10);
	PrintMatrix(input, 1, 10, "Input");
	PrintMatrix(output, 1, 10, "Relu");
}