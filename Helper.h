#pragma once
#include "cuda_runtime.h"
#define cuda_error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }

double random_double();
bool fill_matrix(double *array,int rows,int columns);
double*  copy_matrix_to_device(double* destination,double* host_array, int rows, int columns);
void gpuAssert(cudaError_t code, const char* file, int line);
class Timer
{
private:
	cudaEvent_t start;
	cudaEvent_t stop;
public:
	Timer();
	~Timer();
	void start_count();
	void stop_count();
	float time();
};