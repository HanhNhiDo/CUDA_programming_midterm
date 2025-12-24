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
    assert(h_c[i * N + j] == sum);
    }
}
printf("Correct!\n");
}

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
const uint cRow = blockIdx.y;
const uint cCol = blockIdx.x;

// each warp will calculate 32*TM elements, with 32 being the columnar dim.
const int threadCol = threadIdx.x % BN;
const int threadRow = threadIdx.x / BN;

// allocate space for the current blocktile in SMEM
__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];

// Move blocktile to beginning of A's row and B's column
A += cRow * BM * K;
B += cCol * BN;
C += cRow * BM * N + cCol * BN;

assert(BM * BK == blockDim.x);
assert(BN * BK == blockDim.x);

const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
const uint innerRowA = threadIdx.x / BK;
const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
const uint innerRowB = threadIdx.x / BN;

// allocate thread-local cache for results in registerfile
float threadResults[TM] = {0.0};

// outer loop over block tiles
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the SMEM caches
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    // advance blocktile
    A += BK;
    B += BK * N;

    // calculate per-thread results
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    float tmpB = Bs[dotIdx * BN + threadCol];
    for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
    }
    }
    __syncthreads();
}

// write out the results
for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];
}
}

template<typename T>
__host__ void copyFromHostToDevice(T* h_a, T* h_b, T* d_a, T* d_b, size_t M, size_t N , size_t K) {
size_t a_bytes = sizeof(T) * M * K;
size_t b_bytes = sizeof(T) * K * N;

HIP_CHECK(hipMemcpy(d_a, h_a, a_bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_b, h_b, b_bytes, hipMemcpyHostToDevice));
}

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
dim3 block((BM * BN) / TM);
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

// Launch kernel
gemm_kernel<T, BM, BN, BK, TM><<<grid, block>>>(d_a, d_b, d_c, M, N, K);

// Check launch error and sync error
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

executeKernel<float, 64, 64, 8, 8>(d_a, d_b, d_c, M, N, K);

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