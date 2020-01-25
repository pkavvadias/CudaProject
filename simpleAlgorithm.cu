#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Multiplications.cuh"

__global__ void simple_algorithm(const double* A, double* C, const int rows, const int columns) {

	double element = 0.0;
	const int 	row = blockIdx.y * blockDim.y + threadIdx.y,
	            col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < rows && col < columns)
	{
		for (int k = 0; k < rows; ++k)
		{
			element += A[k * columns + row] * A[k * columns + col];
		}
		C[row * columns + col] = element;
	}

}