#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Multiplications.cuh"


#define TILE_DIM 32	//Tile dimension
#define BLOCK_SIZE_PER_DIM 16	//Block dimension

__global__ void optimized_algorithm(const double* __restrict__ A_d, double* C_d, int ARows, int ACols) {

	double CValue = 0.0;
	//Õðïëïãéóìüò ôùí äåéêôþí ôçò ãñáììÞò êáé ôçò óôÞëçò
	int Row = blockIdx.y * TILE_DIM + threadIdx.y;
	int Col = blockIdx.x * TILE_DIM + threadIdx.x;
	//×ñÞóç êáôá÷ùñçôþí
	int Var0 = blockIdx.y * TILE_DIM + threadIdx.x;
	int Var1 = threadIdx.y * ACols + Var0;
	int Var2 = threadIdx.y * ACols + Col;
	int Var3 = TILE_DIM * ACols;
	int Var4 = ((blockIdx.y * blockDim.y + threadIdx.y) * ACols) + (blockIdx.x * blockDim.x) + threadIdx.x;
	int Var5 = threadIdx.y;
	//×ñÞóç êïéíÞò ìíÞìçò
	__shared__ volatile double As[TILE_DIM][TILE_DIM];
	__shared__ volatile double Bs[TILE_DIM][TILE_DIM];

	int counter = (TILE_DIM + ARows - 1) / TILE_DIM;

	for (int k = 0; k < counter; k++) {
		//Ìçäåíéóìüò ôùí óôïé÷åßùí ôùí tiles ðïõ âñßóêïíôáé åêôïò ïñßùí ôïõ ìçôñþïõ
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
		//Õðïëïãéóìüò åíäéÜìåóùí ôéìþí
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
	//ÁðïèÞêåõóç ôåëéêÞò ôéìÞò
	if (Row < ACols && Col < ACols)
		C_d[Var4] = CValue;
}