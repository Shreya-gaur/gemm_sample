#include <iostream>
#include <cuda_runtime.h>
#include <cuda.h>
#include <chrono>
#include <ctime>
#include <iomanip>

//structure of choices for user

struct Options{
	bool test;
	bool gemm_cpu_basic;
	bool gemm_cpu_outerloop;
	bool gemm_cpu_tiled;
	bool gemm_gpu_globalmem;
	bool gemm_gpu_optimized;

	Options():
		test(true),
		gemm_cpu_basic(false),
		gemm_cpu_outerloop(false),
		gemm_cpu_tiled(false),
		gemm_gpu_globalmem(false),
		gemm_gpu_optimized(false)
	{  }
	
};

// Debug and Helping Functions

int* filler(int* arr, int n){

	for(int i=0; i<n; i++){
		arr[i] = rand() % 10;
	}

	return arr;

}

int* filler_zero(int* arr, int n){

	for(int i=0; i<n; i++){
		arr[i] = 0;
	}

	return arr;

}

void debugPrint(int* arr, int n){
	for(int i=0; i<n; i++){
		printf("%d, ", arr[i]);
	}
	printf("\n");
}

void debugPrint2D(int* arr, int m, int n){
	for(int i=0; i<m; i++){
		for (int j=0; j<n; j++){
			printf("%d, ", arr[i * n + j]);
		}
		printf("\n");
	}
	printf("\n");
}

int compare2D(int* arr1, int* arr2, int dim1, int dim2){
	
	int correct{0};
	for(int i=0; i<dim1; i++){
		for(int j=0; j<dim2; j++){
			if(arr1[i * dim2 + j] ==  arr2[i * dim2 + j]){
				++correct;
			}
		}
	}

	return correct;
}

// Declarations GPU kernels for different functions

extern __global__ void saxpy(int*, int*, int);
extern void matmul_cpu_basic(int* a, int* b, int* c_cpu, int m, int n, int k);
extern void matmul_cpu_outerloop(int* a, int* b, int* c_cpu, int m, int n, int k);
extern void matmul_cpu_tiled(int* a, int* b, int* c_cpu, int m, int n, int k);
extern __global__ void matmul(int* d_a, int* d_b, int* d_c, int m, int n, int k);
extern __global__ void matmul_shr(int* d_a, int* d_b, int* d_c, int m, int n, int k);
