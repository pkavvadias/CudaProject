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

// Declaration of array As on shared memory to store submatrix of A
__shared__ float As[TILE_DIM][TILE_DIM];

// Declaration of array Bs on shared memory to store submatrix of B
__shared__ double Bs[TILE_DIM][TILE_DIM];

// Index of the first submatrix of A
int aBegin = COLUMNS * TILE_DIM * by;

// Index of the last submatrix of A
int aEnd = aBegin + COLUMNS - 1;

// Step size for iteration through submatrices of A
int aStep = TILE_DIM;

// Index of the first submatrix of A(on our case B is transpose of A)
int bBegin = TILE_DIM * bx;

// Step size for iteration through submatrices of B
int bStep = TILE_DIM * COLUMNS;

// Csub is used to store the element of the block sub-matrix
// that is computed by the thread
float Csub = 0;

// Loop over all the sub-matrices of A and B
// required to compute the block sub-matrix
for (int a = aBegin, b = bBegin;
	a <= aEnd;
	a += aStep, b += bStep) {


	// Load the matrices from device memory
	// to shared memory; each thread loads
	// one element of each matrix
	As[ty][tx] = A_d[a + COLUMNS * ty + tx];//fIX
	Bs[ty][tx] = A_d[b + COLUMNS * ty + tx];

	// Synchronize to make sure the matrices are loaded
	__syncthreads();

	// Multiply the two matrices together;
	// each thread computes one element
	// of the block sub-matrix
	for (int k = 0; k < TILE_DIM; ++k)
		Csub += As[ty][ k] * Bs[k][tx];

	// Synchronize to make sure that the preceding
	// computation is done before loading two new
	// sub-matrices of A and B in the next iteration
	__syncthreads();
}

// Write the block sub-matrix to device memory;
// each thread writes one element
int c = COLUMNS * TILE_DIM * by + TILE_DIM * bx;
C_d[c + COLUMNS * ty + tx] = Csub;

}