#include <iostream>
#include "Helper.h"
#include "Multiplications.cuh"

#define ROWS 30
#define COLUMNS 30

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
	 * Menu
	 */

	std::cout << "We have a matrix of "<<ROWS<<" rows and "<<COLUMNS<<" columns and we multiply it with its transpose"<<std::endl;
	std::cout << "Select method of multiplication" << std::endl;
	std::cout << "1.Using cublasDgemm" << std::endl;
	std::cout << "2.Using our algorithm" << std::endl;
	std::cout << "3.Using our optimized algorithm" << std::endl;
	std::cout << "Any other number to exit" << std::endl;

	std::cin.clear();
	std::cin >> selector;

	switch(selector){
	case 1:
		cublass_multiplication(device_A, device_C, ROWS, COLUMNS);
		break;
	case 2:
		// TODO: Simple multiplication algorithm
		break;
	case 3:
		// TODO: Optimized multiplication algorithm
		break;
	}
		
	free(A);
	free(C);
	cuda_error_check(cudaFree(device_A));
	cuda_error_check(cudaFree(device_C));
	exit(EXIT_SUCCESS);
}
