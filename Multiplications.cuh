#pragma once
void cublas_multiplication(const double* A, double* C, const int rows, const int columns);
__global__ void simple_algorithm(const double* A, double* C, const int rows, const int columns);