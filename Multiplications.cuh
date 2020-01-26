#pragma once
/*
 * Declaration of multiplication functions
 */
void cublas_multiplication(const double* A, double* C, const int rows, const int columns);
__global__ void simple_algorithm(const double* A, double* C, const int rows, const int columns);
__global__ void optimized_algorithm(const double* __restrict__ A_d, double* C_d, int ARows, int ACols);