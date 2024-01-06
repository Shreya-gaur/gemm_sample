#include "utils.h"

#define DEBUG
#define TILE_WIDTH 16


__global__
void saxpy(int* d_a, int* d_b, int n){

	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	if (tid < n){
		d_a[tid] = (10 * d_a[tid]) + d_b[tid];		
	}

}

void matmul_cpu_basic(int* a, int* b, int* c_cpu, int m, int n, int k){

	std::clock_t c_start = std::clock();

	for(int row = 0; row < m ; row++){
		for(int col = 0; col < n; col++){
			for(int gemm_k = 0; gemm_k < k; gemm_k++){
				c_cpu[row * n + col] += a[row * k + gemm_k] * b[gemm_k * n + col];
			}
		}
	}

	std::clock_t c_end = std::clock();

	double time_elapsed  = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

	std::cout << "CPU Time Used: "
			  << time_elapsed
			  << " ms\n";
}

void matmul_cpu_outerloop(int* a, int* b, int* c_cpu, int m, int n, int k){

	std::clock_t c_start = std::clock();

	for(int gemm_k = 0; gemm_k < k; gemm_k++){
		for(int col = 0; col < n; col++){
			for(int row = 0; row < m ; row++){
				c_cpu[row * n + col] += a[row * k + gemm_k] * b[gemm_k * n + col];
			}
		}
	}

	std::clock_t c_end = std::clock();

	double time_elapsed  = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

	std::cout << "CPU Time Used: "
			  << time_elapsed
			  << " ms\n";
}

void matmul_cpu_tiled(int* a, int* b, int* c, int m, int n, int k){

	std::clock_t c_start = std::clock();

	int tilesAlongN = (n + TILE_WIDTH - 1) / TILE_WIDTH;
	int tilesAlongM = (m + TILE_WIDTH - 1) / TILE_WIDTH;
	int tilesAlongK = (k + TILE_WIDTH - 1) / TILE_WIDTH;

	int tileId = 0;

	while(tileId < tilesAlongN * tilesAlongM){

		int offsetM = (tileId / tilesAlongN) * TILE_WIDTH;
		int offsetN = (tileId % tilesAlongN) * TILE_WIDTH;

		int rowIdx, colIdx;
		int row, col, gemm_k, subTileK, kcoord;

		for(int gemm_k = 0; gemm_k < tilesAlongK; ++gemm_k){

			for(row = 0; row < TILE_WIDTH; ++row){
				for(col = 0; col < TILE_WIDTH; ++col){

					rowIdx =  row + offsetM;
					colIdx =  col + offsetN;

					if( rowIdx < m && colIdx < n) {

						if(gemm_k == 0) c[rowIdx * n + colIdx] = 0;

						for(subTileK = 0; subTileK < TILE_WIDTH; ++subTileK){
							kcoord = gemm_k * TILE_WIDTH + subTileK;
							if(kcoord < k){
								c[rowIdx * n + colIdx] +=
									a[rowIdx * k + kcoord] * b[kcoord * n + colIdx];
							}

						}

					}

				}
			}

		}

		++tileId;

	}

	std::clock_t c_end = std::clock();

	double time_elapsed  = 1000.0 * (c_end - c_start) / CLOCKS_PER_SEC;

	std::cout << "CPU Time Used: "
			  << time_elapsed
			  << " ms\n";
}



__global__
void matmul(int* d_a, int* d_b, int* d_c, int m, int n, int k){

 	int	row = blockDim.y * blockIdx.y + threadIdx.y;
	int col = blockDim.x * blockIdx.x + threadIdx.x;

	if(row < m && col < n){

		for(int i=0; i < k; i++){

		// Note that the accesses to d_a are not coalesced
		// Each thread picks up a row of matrix a. 
		// Thread 0 and thread 1 for d_a are accessing the first row and the second row
		// There is a stride in between of k elements. ie. k * 4 Bytes
		// For k = 32 the stride is 128 bytes which is the size of one DRAM burst.
		// Coalescing is therefore not perfect.

			d_c[row * n + col] += d_a[row * k + i] * d_b[i * n + col]; 

		}
	}
}

__global__
void matmul_shr(int* d_a, int* d_b, int* d_c, int m, int n, int k){

 	int	offsetM = TILE_WIDTH * blockIdx.y;
	int offsetN = TILE_WIDTH * blockIdx.x;

	int tilesAlongK = (k + TILE_WIDTH - 1) / TILE_WIDTH;

	int Cvalue = 0;

	int* tileCptr = &d_c[offsetM * n + offsetN];

	for(int gemm_k=0 ; gemm_k < tilesAlongK ; ++gemm_k){

		int* tileAptr = &d_a[k * offsetM + gemm_k * TILE_WIDTH]; // col of the A matrix is blockIdx.y * gemm_k]
		int* tileBptr = &d_b[n * gemm_k * TILE_WIDTH + offsetN];

		__shared__ int tileA[TILE_WIDTH][TILE_WIDTH];
		__shared__ int tileB[TILE_WIDTH][TILE_WIDTH];
		
		tileA[threadIdx.y][threadIdx.x] = tileAptr[threadIdx.y * k + threadIdx.x];
		tileB[threadIdx.y][threadIdx.x] = tileBptr[threadIdx.y * n + threadIdx.x];

		__syncthreads();

		if(gemm_k == 0) Cvalue = 0;

		for(int e = 0 ; e < TILE_WIDTH ; ++e)

			Cvalue += tileA[threadIdx.y][e] * tileB[e][threadIdx.x];

		__syncthreads();
	}

	tileCptr[threadIdx.y * n + threadIdx.x] = Cvalue;

}


//int main(int argc, const char** argv){
int main(){

	Options options;
	int choice;

	printf("------------------------------\n"); 
	
	std::cout << "PRESS 1 FOR TEST RUN OF SAXPY\n" 
			<< "PRESS 2 FOR BASIC GEMM ON CPU\n"
			<< "PRESS 3 FOR GEMM ON CPU WITH OUTERLOOP OPTIMIZATION\n"
			<< "PRESS 4 FOR GEMM ON CPU WITH TILING OPTIMIZATION\n"
			<< "PRESS 5 FOR UNOPTIMIZED GEMM FOR GPU\n"
			<< "PRESS 6 FOR OPTIMIZED GEMM FOR GPU\n";
	
	printf("------------------------------\n"); 

	std::cin >> choice;

	switch(choice){
		case 1:
			//Leave as it is for default
			break;
		case 2: 
			options.test = false;
			options.gemm_cpu_basic = true;
			break;
		case 3: 
			options.test = false;
			options.gemm_cpu_outerloop = true;
			break;
		case 4: 
			options.test = false;
			options.gemm_cpu_tiled = true;
			break;
		case 5:
			options.test = false;
			options.gemm_gpu_globalmem = true;
			break;
		case 6:
			options.test = false;
			options.gemm_gpu_optimized = true;
			break;
	}

	if (options.test){
			
		int n;
		
		std::cout << "Input size of A and B (N): " ;

		std::cin >> n;
	
		size_t size = n * sizeof(int);

		cudaError_t cudaStatus;

		int *a, *b;

		a = (int*) malloc(size);
		b = (int*) malloc(size);

		a = filler(a, n);
		b = filler(b, n);
	
		int *d_a, *d_b;

		cudaStatus = cudaMalloc(&d_a, size);

		if(cudaStatus != cudaSuccess){
			std::cout << "d_a malloc failed!";
			return 0;
		}

		cudaStatus = cudaMalloc(&d_b, size);

		if(cudaStatus != cudaSuccess){
			std::cout << "d_b malloc failed!";
			return 0;
		}

		#ifdef DEBUG
		std::cout<< "Input A" << '\n';
		debugPrint(a, n);

		std::cout<< "Input B" << '\n';
		debugPrint(b, n);
		#endif

		cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
		
		dim3 blockDim(128,1,1);
		dim3 gridDim((n + blockDim.x - 1 / blockDim.x), 1, 1);

		saxpy<<<gridDim, blockDim>>>(d_a, d_b, n);

		cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost);

		std::cout << "Test ran successfully" << 'n';

		#ifdef DEBUG
		std::cout<< "Output" << '\n';
		debugPrint(a, n);
		#endif

		cudaFree(d_a);
		cudaFree(d_b);

	}
	
	if(options.gemm_cpu_basic){

		std::cout<< "CPU GEMM Kernel Testing" << "\n";

		int m{32}, n{32}, k{32};
		
		std::cout << "Enter M, N, K: " ;

		std::cin >> m >> n >> k;
	
		size_t size_a = m * k * sizeof(int);
		size_t size_b = k * n * sizeof(int);
		size_t size_c = m * n * sizeof(int);

		int *a, *b, *c_cpu;

		a = (int*) malloc(size_a);
		b = (int*) malloc(size_b);
		c_cpu = (int*) malloc(size_c);

		a = filler(a, m*k);
		b = filler(b, k*n);
		c_cpu = filler_zero(c_cpu, m*n);

		#ifdef DEBUG
		std::cout<< "Input A" << '\n';
		debugPrint2D(a, m, k);
		std::cout<< "Input B" << '\n';
		debugPrint2D(b, n, k);
		#endif
		
		matmul_cpu_basic(a, b, c_cpu, m, n, k);
	
		#ifdef DEBUG
		std::cout<< "Output CPU" << '\n';
		debugPrint2D(c_cpu, m, n);
		#endif

		free(a);
		free(b);
		free(c_cpu);

	}
	
	if(options.gemm_cpu_outerloop){

		std::cout<< "CPU GEMM Kernel Testing with outerloop optimization" << "\n";

		int m{32}, n{32}, k{32};
		
		std::cout << "Enter M, N, K: " ;

		std::cin >> m >> n >> k;
	
		size_t size_a = m * k * sizeof(int);
		size_t size_b = k * n * sizeof(int);
		size_t size_c = m * n * sizeof(int);

		int *a, *b, *c_cpu;

		a = (int*) malloc(size_a);
		b = (int*) malloc(size_b);
		c_cpu = (int*) malloc(size_c);

		a = filler(a, m*k);
		b = filler(b, k*n);
		c_cpu = filler_zero(c_cpu, m*n);

		#ifdef DEBUG
		std::cout<< "Input A" << '\n';
		debugPrint2D(a, m, k);
		std::cout<< "Input B" << '\n';
		debugPrint2D(b, n, k);
		#endif
		
		matmul_cpu_outerloop(a, b, c_cpu, m, n, k);
	
		#ifdef DEBUG
		std::cout<< "Output CPU" << '\n';
		debugPrint2D(c_cpu, m, n);
		#endif

		free(a);
		free(b);
		free(c_cpu);

	}

	if(options.gemm_cpu_tiled){

		std::cout<< "CPU GEMM Kernel Testing with tiled optimization" << "\n";

		int m{32}, n{32}, k{32};
		
		std::cout << "Enter M, N, K: " ;

		std::cin >> m >> n >> k;
	
		size_t size_a = m * k * sizeof(int);
		size_t size_b = k * n * sizeof(int);
		size_t size_c = m * n * sizeof(int);

		int *a, *b, *c_cpu;

		a = (int*) malloc(size_a);
		b = (int*) malloc(size_b);
		c_cpu = (int*) malloc(size_c);

		a = filler(a, m*k);
		b = filler(b, k*n);
		c_cpu = filler_zero(c_cpu, m*n);

		#ifdef DEBUG
		std::cout<< "Input A" << '\n';
		debugPrint2D(a, m, k);
		std::cout<< "Input B" << '\n';
		debugPrint2D(b, n, k);
		#endif
		
		matmul_cpu_tiled(a, b, c_cpu, m, n, k);
	
		#ifdef DEBUG
		std::cout<< "Output CPU" << '\n';
		debugPrint2D(c_cpu, m, n);
		#endif

		free(a);
		free(b);
		free(c_cpu);

	}
	
	if (options.gemm_gpu_globalmem){

		std::cout<< "Basic GEMM Kernel Testing" << "\n";

		int m{32}, n{32}, k{32};
		
		std::cout << "Enter M, N, K: " ;

		std::cin >> m >> n >> k;
	
		size_t size_a = m * k * sizeof(int);
		size_t size_b = k * n * sizeof(int);
		size_t size_c = m * n * sizeof(int);

		cudaError_t cudaStatus;

		int *a, *b, *c, *c_cpu;

		a = (int*) malloc(size_a);
		b = (int*) malloc(size_b);
		c = (int*) malloc(size_c);
		c_cpu = (int*) malloc(size_c);

		a = filler(a, m*k);
		b = filler(b, k*n);
		c = filler_zero(c, m*n);
		c_cpu = filler_zero(c_cpu, m*n);
	
		int *d_a, *d_b, *d_c;

		cudaStatus = cudaMalloc(&d_a, size_a);

		if(cudaStatus != cudaSuccess){
			std::cout << "d_a malloc failed!";
			return -1;
		}

		cudaStatus = cudaMalloc(&d_b, size_b);

		if(cudaStatus != cudaSuccess){
			std::cout << "d_b malloc failed!";
			return -1;
		}

		cudaStatus = cudaMalloc(&d_c, size_c);

		if(cudaStatus != cudaSuccess){
			std::cout << "d_c malloc failed!";
			return -1;
		}

		#ifdef DEBUG
		std::cout<< "Input A" << '\n';
		debugPrint2D(a, m, k);
		std::cout<< "Input B" << '\n';
		debugPrint2D(b, n, k);
		std::cout<< "Input C" << '\n';
		debugPrint2D(c, m, n);
		#endif

		matmul_cpu_basic(a, b ,c_cpu, m, n, k);

		cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(d_c, c, size_c, cudaMemcpyHostToDevice);
		
		dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
		dim3 gridDim(((m + blockDim.x - 1)/ blockDim.x), ((n + blockDim.y - 1 )/blockDim.y));

		matmul<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);

		cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

		int correct = compare2D(c, c_cpu, m, n);
		std::cout << "Accuracy: "
				  << correct << "/" << m*n 
				  << '\n';

		#ifdef DEBUG
		std::cout<< "Output CPU" << '\n';
		debugPrint2D(c_cpu, m, n);
		std::cout<< "Output GPU" << '\n';
		debugPrint2D(c, m, n);
		#endif

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		free(a);
		free(b);
		free(c);
		free(c_cpu);

	}

	if (options.gemm_gpu_optimized){

		std::cout<< "Optimized GEMM Kernel Testing" << "\n";

		int m{32}, n{32}, k{32};
		
		std::cout << "Enter M, N, K: " ;

		std::cin >> m >> n >> k;
	
		size_t size_a = m * k * sizeof(int);
		size_t size_b = k * n * sizeof(int);
		size_t size_c = m * n * sizeof(int);

		cudaError_t cudaStatus;

		int *a, *b, *c, *c_cpu;

		a = (int*) malloc(size_a);
		b = (int*) malloc(size_b);
		c = (int*) malloc(size_c);
		c_cpu = (int*) malloc(size_c);

		a = filler(a, m*k);
		b = filler(b, k*n);
		c = filler_zero(c, m*n);
		c_cpu = filler_zero(c_cpu, m*n);
	
		int *d_a, *d_b, *d_c;

		cudaStatus = cudaMalloc(&d_a, size_a);

		if(cudaStatus != cudaSuccess){
			std::cout << "d_a malloc failed!";
			return -1;
		}

		cudaStatus = cudaMalloc(&d_b, size_b);

		if(cudaStatus != cudaSuccess){
			std::cout << "d_b malloc failed!";
			return -1;
		}

		cudaStatus = cudaMalloc(&d_c, size_c);

		if(cudaStatus != cudaSuccess){
			std::cout << "d_c malloc failed!";
			return -1;
		}

		#ifdef DEBUG
		std::cout<< "Input A" << '\n';
		debugPrint2D(a, m, k);
		std::cout<< "Input B" << '\n';
		debugPrint2D(b, n, k);
		std::cout<< "Input C" << '\n';
		debugPrint2D(c, m, n);
		#endif

		matmul_cpu_basic(a, b ,c_cpu, m, n, k);

		cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice);
		cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice);
		cudaMemcpy(d_c, c, size_c, cudaMemcpyHostToDevice);
		
		dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
		dim3 gridDim(((m + blockDim.y - 1)/ blockDim.y), ((n + blockDim.x - 1 )/blockDim.x));

		matmul_shr<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);

		cudaMemcpy(c, d_c, size_c, cudaMemcpyDeviceToHost);

		int correct = compare2D(c, c_cpu, m, n);
		std::cout << "Accuracy: "
				  << correct << "/" << m*n 
				  << '\n';

		#ifdef DEBUG
		std::cout<< "Output CPU" << '\n';
		debugPrint2D(c_cpu, m, n);
		std::cout<< "Output GPU" << '\n';
		debugPrint2D(c, m, n);
		#endif

		cudaFree(d_a);
		cudaFree(d_b);
		cudaFree(d_c);
		free(a);
		free(b);
		free(c);
		free(c_cpu);

	}

	return 0;

}
