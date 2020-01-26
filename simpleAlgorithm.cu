#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Multiplications.cuh"

/*
 * Naive algorithm for matrix multiplication
 */
__global__ void simple_algorithm(const double* A, double* C, const int rows, const int columns) {

	const int 	row = blockIdx.y * blockDim.y + threadIdx.y,//Calculates row inside block
	            col = blockIdx.x * blockDim.x + threadIdx.x;//Calculates column inside block
	if (row < rows && col < columns)//Checking that rows and columns are within range of matrix A
	{	//Stores the element calculated
		double element = 0.0;
		for (int k = 0; k < rows; k++)
		{
			element += A[k * columns + row] * A[k * columns + col];//Multiplication of A with its transpose
		}
		C[row * columns + col] = element;//Stores each element on its corresponding position inside C matrix
	}

}