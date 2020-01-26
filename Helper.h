#pragma once
#include "cuda_runtime.h"
#define cuda_error_check(ans) { gpuAssert((ans), __FILE__, __LINE__); }//Error checking macro

#define TILE_DIM 16 //Tile Dimension
/*
 * Number of rows and columns of matrix A
 */
#define ROWS 1200	
#define COLUMNS 100
/*
 * Number of threads for naive implementation
 */
#define THREADS 16

double random_double();//Random double generator
bool fill_matrix(double *array,int rows,int columns);//Fills matrix with random doubles
double*  copy_matrix_to_device(double* destination,double* host_array, int rows, int columns);//Copies matrix from host to device
void gpuAssert(cudaError_t code, const char* file, int line);//Error checking function declaration

/*
 * Timer declaration
 */
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