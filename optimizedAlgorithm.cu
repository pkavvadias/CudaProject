#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Multiplications.cuh"
#include "Helper.h"

	//Tile dimension
//#define BLOCK_SIZE_PER_DIM 16	//Block dimension

__global__ void optimized_algorithm(const double* __restrict__ A_d, double* C_d, int ARows, int ACols) {
	/**
	double CValue = 0.0;
	int Row = blockIdx.y * TILE_DIM + threadIdx.y;
	int Col = blockIdx.x * TILE_DIM + threadIdx.x;
	int Var0 = blockIdx.y * TILE_DIM + threadIdx.x;
	int Var1 = threadIdx.y * ACols + Var0;
	int Var2 = threadIdx.y * ACols + Col;
	int Var3 = TILE_DIM * ACols;
	int Var4 = ((blockIdx.y * blockDim.y + threadIdx.y) * ACols) + (blockIdx.x * blockDim.x) + threadIdx.x;
	int Var5 = threadIdx.y;
	__shared__ volatile double As[TILE_DIM][TILE_DIM];
	__shared__ volatile double Bs[TILE_DIM][TILE_DIM];

	int counter = (TILE_DIM + ARows - 1) / TILE_DIM;

	for (int k = 0; k < counter; k++) {

		if (Var5 < ARows && Var0 < ACols)
			As[threadIdx.x][threadIdx.y] = A_d[k * Var3 + Var1];
		else
			As[threadIdx.x][threadIdx.y] = 0.0;

		if (Var5 < ARows && Col < ACols)
			Bs[threadIdx.y][threadIdx.x] = A_d[k * Var3 + Var2];
		else
			Bs[threadIdx.y][threadIdx.x] = 0.0;
		Var5 = k * TILE_DIM + threadIdx.y;
		__syncthreads();

		CValue = CValue + As[threadIdx.y][0] * Bs[0][threadIdx.x]
			+ As[threadIdx.y][1] * Bs[1][threadIdx.x]
			+ As[threadIdx.y][2] * Bs[2][threadIdx.x]
			+ As[threadIdx.y][3] * Bs[3][threadIdx.x]
			+ As[threadIdx.y][4] * Bs[4][threadIdx.x]
			+ As[threadIdx.y][5] * Bs[5][threadIdx.x]
			+ As[threadIdx.y][6] * Bs[6][threadIdx.x]
			+ As[threadIdx.y][7] * Bs[7][threadIdx.x]
			+ As[threadIdx.y][8] * Bs[8][threadIdx.x]
			+ As[threadIdx.y][9] * Bs[9][threadIdx.x]
			+ As[threadIdx.y][10] * Bs[10][threadIdx.x]
			+ As[threadIdx.y][11] * Bs[11][threadIdx.x]
			+ As[threadIdx.y][12] * Bs[12][threadIdx.x]
			+ As[threadIdx.y][13] * Bs[13][threadIdx.x]
			+ As[threadIdx.y][14] * Bs[14][threadIdx.x]
			+ As[threadIdx.y][15] * Bs[15][threadIdx.x];
		__syncthreads();
	}

	if (Row < ACols && Col < ACols)
		C_d[Var4] = CValue;
	*/
	/**
	__shared__  double ds_M[TILE_DIM][TILE_DIM];
	__shared__  double ds_N[TILE_DIM][TILE_DIM];
	int bx = blockIdx.x, by = blockIdx.y,
		tx = threadIdx.x, ty = threadIdx.y,
		Row = by * TILE_DIM + ty,
		Col = bx * TILE_DIM + tx;
	double Pvalue = 0;
	//int counter = (TILE_DIM + ARows - 1) / TILE_DIM;

	//for (int m = 0; m < counter; m++) {
	//for (int m = 0; m < (TILE_DIM*ARows-1) / TILE_DIM + 1; ++m) {
	for (int m = 0; m < ACols/TILE_DIM; ++m) {

		if (Row < ARows && m * TILE_DIM + tx < ACols)
			ds_M[ty][tx] = A_d[(by*TILE_DIM+tx)+(m*TILE_DIM+ty)*ACols];
		else
			ds_M[ty][tx] = 0;
		if (Col < ACols && m * TILE_DIM + ty < ARows)
			ds_N[ty][tx] = A_d[(m * TILE_DIM + ty) * ACols + Col];
		else
			ds_N[ty][tx] = 0;

		
		__syncthreads();
		/**
		for (int k = 0; k < TILE_DIM; ++k) {
			Pvalue += ds_M[ty][k] * ds_N[k][tx];
		}
		*/
	/**
		Pvalue = Pvalue + ds_M[threadIdx.y][0] * ds_N[0][threadIdx.x]
			+ ds_M[threadIdx.y][1] * ds_N[1][threadIdx.x]
			+ ds_M[threadIdx.y][2] * ds_N[2][threadIdx.x]
			+ ds_M[threadIdx.y][3] * ds_N[3][threadIdx.x]
			+ ds_M[threadIdx.y][4] * ds_N[4][threadIdx.x]
			+ ds_M[threadIdx.y][5] * ds_N[5][threadIdx.x]
			+ ds_M[threadIdx.y][6] * ds_N[6][threadIdx.x]
			+ ds_M[threadIdx.y][7] * ds_N[7][threadIdx.x]
			+ ds_M[threadIdx.y][8] * ds_N[8][threadIdx.x]
			+ ds_M[threadIdx.y][9] * ds_N[9][threadIdx.x]
			+ ds_M[threadIdx.y][10] * ds_N[10][threadIdx.x]
			+ ds_M[threadIdx.y][11] * ds_N[11][threadIdx.x]
			+ ds_M[threadIdx.y][12] * ds_N[12][threadIdx.x]
			+ ds_M[threadIdx.y][13] * ds_N[13][threadIdx.x]
			+ ds_M[threadIdx.y][14] * ds_N[14][threadIdx.x]
			+ ds_M[threadIdx.y][15] * ds_N[15][threadIdx.x];
		__syncthreads();
		
	}
		
	if (Row < ARows && Col < ACols)
		C_d[Row * ACols + Col] = Pvalue;
		*/

#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]

// Block index
int bx = blockIdx.x;
int by = blockIdx.y;

// Thread index
int tx = threadIdx.x;
int ty = threadIdx.y;

// Declaration of the shared memory array As used to
// store the sub-matrix of A
__shared__ float As[TILE_DIM][TILE_DIM];

// Declaration of the shared memory array Bs used to
// store the sub-matrix of B
__shared__ float Bs[TILE_DIM][TILE_DIM];

// Index of the first sub-matrix of A processed by the block
int aBegin = COLUMNS * TILE_DIM * by;

// Index of the last sub-matrix of A processed by the block
int aEnd = aBegin + COLUMNS - 1;

// Step size used to iterate through the sub-matrices of A
int aStep = TILE_DIM;

// Index of the first sub-matrix of B processed by the block
int bBegin = TILE_DIM * bx;

// Step size used to iterate through the sub-matrices of B
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
	AS(ty, tx) = A_d[a + COLUMNS * ty + tx];//fIX
	BS(ty, tx) = A_d[b + COLUMNS * ty + tx];

	// Synchronize to make sure the matrices are loaded
	__syncthreads();

	// Multiply the two matrices together;
	// each thread computes one element
	// of the block sub-matrix
	for (int k = 0; k < TILE_DIM; ++k)
		Csub += AS(ty, k) * BS(k, tx);

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