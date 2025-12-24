#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <cstring>

#define BM 128
#define BN 128
#define BK 16   
#define TM 8
#define TN 8
#define NUM_THREADS 256

#define HIP_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP Error: %s at line %d\n", hipGetErrorString(err), __LINE__); \
        exit(1); \
    } \
} while(0)

__global__ void sgemm_best_kernel(int M, int N, int K, const float * __restrict__ A, const float * __restrict__ B, float * __restrict__ C) {
    __shared__ float s_A[BK][BM]; 
    __shared__ float s_B[BK][BN]; 

    float threadResults[TM * TN] = {0.0f};
    float reg_M[TM];
    float reg_N[TN];

    int tid = threadIdx.x;
    int bx = blockIdx.x; 
    int by = blockIdx.y;

    int globalRow = by * BM;
    int globalCol = bx * BN;

    int tid_a = tid;       
    int a_stride = 256;    
    
    int tid_b = tid;
    int b_stride = 256;

    const float* A_ptr = A + globalRow * K; 
    const float* B_ptr = B + globalCol;     

    for (int k = 0; k < K; k += BK) {
        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int load_idx = tid_a + i * a_stride; 
            int row_in_tile = load_idx / 4;      
            int vec_col_in_tile = load_idx % 4;  
            int col_in_tile = vec_col_in_tile * 4; 

            float4 loaded_val = *reinterpret_cast<const float4*>(&A_ptr[row_in_tile * K + (k + col_in_tile)]);

            s_A[col_in_tile + 0][row_in_tile] = loaded_val.x;
            s_A[col_in_tile + 1][row_in_tile] = loaded_val.y;
            s_A[col_in_tile + 2][row_in_tile] = loaded_val.z;
            s_A[col_in_tile + 3][row_in_tile] = loaded_val.w;
        }

        #pragma unroll
        for (int i = 0; i < 2; i++) {
            int load_idx = tid_b + i * b_stride;
            int row_in_tile = load_idx / 32;       
            int vec_col_in_tile = load_idx % 32;   
            int col_in_tile = vec_col_in_tile * 4; 

            float4 loaded_val = *reinterpret_cast<const float4*>(&B_ptr[(k + row_in_tile) * N + col_in_tile]);
            reinterpret_cast<float4*>(&s_B[row_in_tile][col_in_tile])[0] = loaded_val;
        }

        __syncthreads();

        int ty = tid / (BN / TN); 
        int tx = tid % (BN / TN); 

        #pragma unroll
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                reg_M[i] = s_A[dotIdx][ty * TM + i];
            }
            #pragma unroll
            for (int j = 0; j < TN; ++j) {
                reg_N[j] = s_B[dotIdx][tx * TN + j];
            }
            #pragma unroll
            for (int i = 0; i < TM; ++i) {
                #pragma unroll
                for (int j = 0; j < TN; ++j) {
                    threadResults[i * TN + j] += reg_M[i] * reg_N[j];
                }
            }
        }
        __syncthreads();
    }

    int ty = tid / (BN / TN);
    int tx = tid % (BN / TN);
    
    int row = globalRow + ty * TM;
    int col = globalCol + tx * TN;

    #pragma unroll
    for (int i = 0; i < TM; ++i) {
        #pragma unroll
        for (int j = 0; j < TN; ++j) {
            if (row + i < M && col + j < N) {
                C[(row + i) * N + (col + j)] = threadResults[i * TN + j];
            }
        }
    }
}

void parseArgs(int argc, char* argv[], int &M, int &N, int &K) {
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) M = atoi(argv[++i]);
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) N = atoi(argv[++i]);
        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc) K = atoi(argv[++i]);
    }
}

int main(int argc, char* argv[]) {
    int M = 1024, N = 1024, K = 1024;
    parseArgs(argc, argv, M, N, K);

    std::cout << "Best GEMM Config: M=" << M << ", N=" << N << ", K=" << K << std::endl;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N);

    srand(time(NULL));
    float rand_max_f = static_cast<float>(RAND_MAX);
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(rand()) / rand_max_f;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(rand()) / rand_max_f;

    float *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, size_A));
    HIP_CHECK(hipMalloc(&d_B, size_B));
    HIP_CHECK(hipMalloc(&d_C, size_C));

    HIP_CHECK(hipMemcpy(d_A, h_A.data(), size_A, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), size_B, hipMemcpyHostToDevice));

    dim3 block(NUM_THREADS);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    sgemm_best_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    int n_iter = 20; 
    for (int i = 0; i < n_iter; i++) {
        sgemm_best_kernel<<<grid, block>>>(M, N, K, d_A, d_B, d_C);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float milliseconds = 0;
    HIP_CHECK(hipEventElapsedTime(&milliseconds, start, stop));
    float avg_time = milliseconds / n_iter;
    
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double gflops = (flops * 1e-9) / (avg_time * 1e-3);

    std::cout << "Time taken for GEMM: " << avg_time << " ms" << std::endl;
    std::cout << "Performance: " << gflops << " GFLOP/s" << std::endl;

    HIP_CHECK(hipFree(d_A)); 
    HIP_CHECK(hipFree(d_B)); 
    HIP_CHECK(hipFree(d_C));
    HIP_CHECK(hipEventDestroy(start)); 
    HIP_CHECK(hipEventDestroy(stop));

    return 0;
}