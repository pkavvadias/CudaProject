#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Multiplications.cuh"
#include "Helper.h"

__global__ void optimized_algorithm(const double* __restrict__ A_d, double* C_d, int ARows, int ACols) {

// Block index
int bx = blockIdx.x;
int by = blockIdx.y;

// Thread index
int tx = threadIdx.x;
int ty = threadIdx.y;

// Declaration of array As on shared memory to store submatrix of matrix A(On our case A=transpose(A))
__shared__ float As[TILE_DIM][TILE_DIM];

// Index of the first submatrix of A 
int aBegin = ACols * TILE_DIM * by;

// Index of the last submatrix of A
int aEnd = aBegin + ACols - 1;

// Step size for iteration through submatrices of A
int aStep = TILE_DIM;

//Stores the element calculated by the thread
float Csub = 0;

// Loops through all the submatrices of A required to compute the block submatrix
#pragma unroll
for (int a = aBegin;
	a <= aEnd;
	a += aStep) {


	// Load the matrices from device memory to shared memory
	As[ty][tx] = A_d[a + ACols * ty + tx];//Loads A transpose matrix on As

	// Barrier waiting until all threads finished loading
	__syncthreads();

	// Multiplies the two matrices
	#pragma unroll
	for (int k = 0; k < TILE_DIM; ++k)
		Csub += As[ty][ k] * As[k][ty];

	// Waits until all threads finished calculations finished before going to next iteration
	__syncthreads();
}

// Write the block submatrix to device global memory
int c = ACols * TILE_DIM * by + TILE_DIM * bx;
C_d[c + ACols * ty + tx] = Csub;

}