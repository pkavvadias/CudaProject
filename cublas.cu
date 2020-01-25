#include "cuda_runtime.h"
#include "helper.h"
#include <iostream>
#include "cublas_v2.h"

void cublas_multiplication(const double* A, double* C, const int rows, const int columns) {
	Timer t;
	/*
	 * Cublass uses column-major implementation contrary to c/c++ default row-major
	 * For reference about mn,k,lda,ldb,ldc,alpha,beta official reference here:
	 * http://www.netlib.org/blas/dgemm.f
	 */
	int m = columns, n = columns, k = rows;
	int lda = rows, ldb = rows, ldc = columns;
	// C = alpha*A*B + beta*C
	const double alpha = 1, beta = 0;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasStatus_t status;
	t.start_count();

	status=cublasDgemm(handle,
		CUBLAS_OP_T,  // A**T
		CUBLAS_OP_N,  // A 
		m, n, k, &alpha,
		A, lda,
		A, ldb,
		&beta, C, ldc);
	t.stop_count();
	if(status!=CUBLAS_STATUS_SUCCESS)
	{
		std::cout << "Error in cublas operation";
		exit(EXIT_FAILURE);
	}
	std::cout << "Time elapsed to multiply using cublasDgemm is " << t.time()<<" ms"<< std::endl<<std::endl;
	cublasDestroy(handle);
}
