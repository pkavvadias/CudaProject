#include<conio.h>
#include<cstdlib>
#include "Helper.h"
#include <iostream>
#include "cuda_runtime.h"
#include <cublas_v2.h>

/*
 * Function to get random values of type double using rand()
 */
double random_double() {
	double rnd;
	int rmin = -100000, rmax = 100000;

	rnd = (double)rand() / RAND_MAX;
	rnd = rmin + rnd * (rmax - rmin);

	return rnd;
}
/*
 * Fills matrix with random doubles
 */
bool fill_matrix(double* array, int rows, int columns)
{
	for(int i=0;i<rows;i++)
	{
		for(int j=0;j<columns;j++)
		{
			*(array + i * columns + j) = random_double();
		}
	}
	return true;
}
/*
 * Allocates space to device and copies matrix from host
 */
double* copy_matrix_to_device(double* destination, double* host_array, int rows, int columns)
{
	cuda_error_check(cudaMalloc(&destination, rows * columns * sizeof(double)));//Allocate space on device
	cuda_error_check(cudaMemcpy(destination, host_array, rows * columns * sizeof(double), cudaMemcpyHostToDevice));//Copy matrix A from host to device
	return destination;
}
/*
 * Definitions for the timer used to time all cuda calculations
 */
Timer::Timer()
{
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
}
Timer::~Timer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
}
void Timer::start_count()
{
	cudaEventRecord(start, 0);
}
void Timer::stop_count()
{
	cudaEventRecord(stop, 0);
}
float Timer::time()
{
	float elapsed;
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsed, start, stop);
	return elapsed;
}
/*
 * Function for error checking on cuda operations
 */
inline void gpuAssert(cudaError_t code, const char* file, int line) {
	if (code != cudaSuccess) {
		fprintf(stderr, "Error: %s %s %d\n", cudaGetErrorString(code), file, line);
		exit(EXIT_FAILURE);
	}
}