#include <iostream>
#include "Helper.h"
#include "Multiplications.cuh"
#include "device_launch_parameters.h"
#include "cuda_runtime.h"
#define ROWS 300
#define COLUMNS 200
#define THREADS 32

/*For optimized algorithm*/
#define TILE_DIM 32	//Tile dimension
#define BLOCK_SIZE_PER_DIM 16	//Block dimension
int main(int argc, char* argv[])
{
	int selector;
	double* device_A=NULL, *device_C=NULL;//Declare pointers for memory on GPU

	/*
	 * Declares A Matrix
	 */
	double *A = (double*)malloc(ROWS * COLUMNS * sizeof(double));

	/*
	 * Declares C Matrix.C stores the result of multiplication A**T*A so rows of C=columns of C=columns of A
	 */
	double* C = (double*)malloc(COLUMNS * COLUMNS * sizeof(double));

	/*
	 * Now we fill matrix A with random doubles
	 */
	fill_matrix(A, ROWS, COLUMNS);

	/*
	 * Matrices A and C are copied to device
	 */
	device_A=copy_matrix_to_device(device_A,A, ROWS, COLUMNS);
	device_C=copy_matrix_to_device(device_C, C, COLUMNS, COLUMNS);

	/*
	 * Create cuda grid and block
	 */
	dim3 block(THREADS, THREADS);
	dim3 grid(ceil(((float)ROWS) / block.x), ceil(((float)COLUMNS) / block.y));

	/*
	 *Grid and block for the optimized algorithm
	 */
	/**
	unsigned int numBlocksX = (ROWS - 1) / BLOCK_SIZE_PER_DIM + 1;
	unsigned int numBlocksY = (COLUMNS - 1) / BLOCK_SIZE_PER_DIM + 1;
	dim3 dimGridOpt(numBlocksX, numBlocksY, 1);
	dim3 dimBlockOpt(BLOCK_SIZE_PER_DIM, BLOCK_SIZE_PER_DIM, 1);
	*/
	dim3 dimGridOpt(ceil((float)COLUMNS / TILE_DIM), ceil((float)ROWS / TILE_DIM),1);
	dim3 dimBlockOpt(TILE_DIM, TILE_DIM, 1);
	/*
	 * Initialize timer
	 */
	Timer t;
	
	/*
	 * Menu
	 */
	selector = 1;
	std::cout << "We have a matrix of " << ROWS << " rows and " << COLUMNS << " columns and we multiply it with its transpose" << std::endl;
	while (selector > 0 && selector < 4) {		
		std::cout << "Select method of multiplication" << std::endl;
		std::cout << "1.Using cublasDgemm" << std::endl;
		std::cout << "2.Using our algorithm" << std::endl;
		std::cout << "3.Using our optimized algorithm" << std::endl;
		std::cout << "Any other number to exit" << std::endl;

		std::cin.clear();
		std::cin >> selector;

		switch (selector) {
		case 1:
			cublas_multiplication(device_A, device_C, ROWS, COLUMNS);
			break;
		case 2:
			t.start_count();
			simple_algorithm << <grid, block >> > (device_A, device_C, ROWS, COLUMNS);
			t.stop_count();
			std::cout << "Time elapsed to multiply using our simple algorithm is " << t.time() << " ms" << std::endl<<std::endl;
			//std::cout << cudaGetLastError();
			break;
		case 3:
			t.start_count();
			optimized_algorithm << <dimGridOpt, dimBlockOpt >> > (device_A, device_C, ROWS, COLUMNS);
			t.stop_count();
			std::cout << "Time elapsed to multiply using our optimized algorithm is " << t.time() << " ms" << std::endl << std::endl;
			break;
			break;
		}
	}
	free(A);
	free(C);
	cuda_error_check(cudaFree(device_A));
	cuda_error_check(cudaFree(device_C));
	exit(EXIT_SUCCESS);
}
