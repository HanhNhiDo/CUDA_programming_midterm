#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h> 
#include <iostream>
#include <tuple>
#include <string>

using namespace std; 

// Macro kiểm tra lỗi HIP chuẩn
#define HIP_CHECK(command) { \
hipError_t status = command; \
if (status != hipSuccess) { \
    fprintf(stderr, "Error: HIP reports %s at line %d\n", hipGetErrorString(status), __LINE__); \
    std::exit(EXIT_FAILURE); \
} \
}

template<typename T>
__host__ void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
    T sum = 0;
    for (int k = 0; k < K; k++) {
        sum += h_a[i * K + k] * h_b[k * N + j];
    }
    // Lưu ý: so sánh float bằng == có thể không chính xác tuyệt đối
    assert(h_c[i * N + j] == sum);
    }
}
printf("Correct!\n");
}

// BM = 128, BN = 128, BK = 16, TM = 8, TN = 8
template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
const uint totalResultsBlocktile = BM * BN; // 128 * 128
// A thread is responsible for calculating TM*TN elements in the blocktile
const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN); // 128 * 128 / (8 * 8) = 256

// ResultsPerBlock / ResultsPerThread == ThreadsPerBlock
assert(numThreadsBlocktile == blockDim.x);

// Calculate thread local index within a block with respect to matrix C
const int threadCol = threadIdx.x % (BN / TN);
const int threadRow = threadIdx.x / (BN / TN);

// allocate space for the current blocktile in smem
__shared__ float As[BM * BK]; // 128 * 16
__shared__ float Bs[BK * BN]; // 16 * 128

// Move blocktile to beginning of A's row and B's column
A += blockIdx.y * BM * K;
B += blockIdx.x * BN;
C += blockIdx.y * BM * N + blockIdx.x * BN;

// calculating the local indices w.r.t matrix A that this thread will load into SMEM
const uint innerRowA = threadIdx.x / BK;
const uint innerColA = threadIdx.x % BK;

// calculates the number of rows of As that are being loaded in a single step
// by a single block
const uint strideA = numThreadsBlocktile / BK;    // 256/16 = 16

// calculating the local indices w.r.t matrix B that this thread will load into SMEM
const uint innerRowB = threadIdx.x / BN;
const uint innerColB = threadIdx.x % BN;
const uint strideB = numThreadsBlocktile / BN;    // 256/128 = 2

// allocate thread-local cache for results in registerfile
float threadResults[TM * TN] = {0.0};
// register caches for As and Bs
float regM[TM] = {0.0};
float regN[TN] = {0.0};

// outer-most loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
    As[(innerRowA + loadOffset) * BK + innerColA] =
        A[(innerRowA + loadOffset) * K + innerColA];
    }
    for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
    Bs[(innerRowB + loadOffset) * BN + innerColB] =
        B[(innerRowB + loadOffset) * N + innerColB];
    }
    __syncthreads();

    // advance blocktile
    A += BK;     // move BK columns to right
    B += BK * N; // move BK rows down

    // calculate per-thread results
    // sequential reduction
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // block into registers
    for (uint i = 0; i < TM; ++i) {
        regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
    }
    for (uint i = 0; i < TN; ++i) {
        regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
    }
    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
        threadResults[resIdxM * TN + resIdxN] +=
            regM[resIdxM] * regN[resIdxN];
        }
    }
    }
    __syncthreads();
}

// write out the results
for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
    C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] = threadResults[resIdxM * TN + resIdxN];
    }
}
}

template<typename T>
__host__ void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N , size_t K) {
size_t a_bytes = sizeof(T) * M * K;
size_t b_bytes = sizeof(T) * K * N;

HIP_CHECK(hipMemcpy(d_a, h_a, a_bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_b, h_b, b_bytes, hipMemcpyHostToDevice));
}

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM, const uint TN>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
dim3 block((BM * BN) / (TM * TN), 1, 1);
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

// Launch kernel
gemm_kernel<T, BM, BN, BK, TM, TN><<<grid, block>>>(d_a, d_b, d_c, M, N, K);

HIP_CHECK(hipGetLastError());
HIP_CHECK(hipDeviceSynchronize());
}

template<typename T>
__host__ void copyFromDeviceToHost(T* d_c, T* h_c, size_t M, size_t N) {
size_t c_bytes = sizeof(T) * M * N;
HIP_CHECK(hipMemcpy(h_c, d_c, c_bytes, hipMemcpyDeviceToHost));
}

template<typename T>
__host__ void deallocateMemory(T* d_a, T* d_b, T* d_c) {
HIP_CHECK(hipFree(d_a));
HIP_CHECK(hipFree(d_b));
HIP_CHECK(hipFree(d_c));
}

__host__ void cleanUpDevice() {
HIP_CHECK(hipDeviceReset());
}

__host__ std::tuple<int, int, int> parseCmdLineArgs(int argc, char *argv[]) {
int M = 1024;
int N = 1024;
int K = 1024;

for (int i = 1; i < argc; i++){
    std::string option(argv[i]);
    if (i + 1 >= argc) break; 
    std::string value(argv[i+1]);
    i++;
    if (option.compare("-m") == 0) M = std::stoi(value);
    else if (option.compare("-n") == 0) N = std::stoi(value);
    else if (option.compare("-k") == 0) K = std::stoi(value);
}
return {M, N, K};
}

int main(int argc, char *argv[]) {
std::tuple<int, int, int>parsedCmdLineArgsTuple = parseCmdLineArgs(argc, argv);
int M = std::get<0>(parsedCmdLineArgsTuple);
int N = std::get<1>(parsedCmdLineArgsTuple);
int K = std::get<2>(parsedCmdLineArgsTuple);

float* h_a = (float*)malloc(M * K * sizeof(float));
float* h_b = (float*)malloc(K * N * sizeof(float));
float* h_c = (float*)malloc(M * N * sizeof(float));

// initialize
for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < K; j++) {
    h_a[i * K + j] = rand() % 10;
    }
}

for (size_t i = 0; i < K; i++) {
    for (size_t j = 0; j < N; j++) {
    h_b[i * N + j] = rand() % 10;
    }
}

// allocate memory on device side
float *d_a, *d_b, *d_c;
HIP_CHECK(hipMalloc((void **)&d_a, M * K * sizeof(float)));
HIP_CHECK(hipMalloc((void **)&d_b, K * N * sizeof(float)));
HIP_CHECK(hipMalloc((void **)&d_c, M * N * sizeof(float)));

copyFromHostToDevice<float>(h_a, h_b, d_a, d_b, M, N, K);

hipEvent_t start, stop;
float time;
HIP_CHECK(hipEventCreate(&start));
HIP_CHECK(hipEventCreate(&stop));

HIP_CHECK(hipEventRecord( start, 0 ));

// Execute kernel với cấu hình tiling:
// BM=128, BN=128, BK=16, TM=8, TN=8
executeKernel<float, 128, 128, 16, 8, 8>(d_a, d_b, d_c, M, N, K);

HIP_CHECK(hipEventRecord( stop, 0 ));
HIP_CHECK(hipEventSynchronize( stop ));

HIP_CHECK(hipEventElapsedTime( &time, start, stop ));
printf("Time taken for GEMM: %f ms\n", time);

HIP_CHECK(hipEventDestroy( start ));
HIP_CHECK(hipEventDestroy( stop ));

cout << "Performance: " << 2LL*M*N*K/(time * 1e-3 * 1e9) << " GFLOP/s\n";

copyFromDeviceToHost<float>(d_c, h_c, M, N);
verifyResult<float>(h_a, h_b, h_c, M, N, K);
deallocateMemory<float>(d_a, d_b, d_c);
cleanUpDevice();

// Free host memory
free(h_a);
free(h_b);
free(h_c);

return 0;
}