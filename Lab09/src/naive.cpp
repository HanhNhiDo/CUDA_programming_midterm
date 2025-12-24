#include <hip/hip_runtime.h>
#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <tuple>
#include <string>
#include <cstdio>

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
__global__ void matmul_kernel(const T *a, const T *b, T *c, int M, int N, int K) {
int col = blockIdx.x * 32 + (threadIdx.x % 32);
int row = blockIdx.y * 32 + (threadIdx.x / 32);
if (row < M && col < N) {
    c[row * N + col] = 0;
    for (int k = 0; k < K; ++k) {
    c[row * N + col] += a[row * K + k] * b[k * N + col];
    }
}
}

template<typename T>
__host__ void verifyResult(T *a, T *b, T *c, int M, int N, int K) {
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
    T sum = 0;
    for (int k = 0; k < K; k++) {
        sum += a[i * K + k] * b[k * N + j];
    }
    assert(c[i * N + j] == sum);
    }
}
cout << "Result is correct!\n";
}

template<typename T>
__host__ void copyFromHostToDevice(T *h_a, T *h_b, T *d_a, T *d_b, int M, int N, int K) {
size_t a_bytes = M * K * sizeof(T);
size_t b_bytes = K * N * sizeof(T);
HIP_CHECK(hipMemcpy(d_a, h_a, a_bytes, hipMemcpyHostToDevice));
HIP_CHECK(hipMemcpy(d_b, h_b, b_bytes, hipMemcpyHostToDevice));
}

template<typename T>
__host__ void executeKernel(T *d_a, T *d_b, T *d_c, int M, int N, int K) {
int block_dim = 32;
dim3 block(block_dim * block_dim);
dim3 grid((N + block_dim - 1) / block_dim, (M + block_dim - 1) / block_dim);

matmul_kernel<T><<<grid, block>>>(d_a, d_b, d_c, M, N, K);

// Kiểm tra lỗi kernel launch
HIP_CHECK(hipGetLastError());

// Kiểm tra lỗi synchronize
HIP_CHECK(hipDeviceSynchronize());
}

template<typename T>
__host__ void copyFromDeviceToHost(T *d_c, T *h_c, int M, int N) {
size_t bytes = M * N * sizeof(T);
HIP_CHECK(hipMemcpy(h_c, d_c, bytes, hipMemcpyDeviceToHost));
}

template<typename T>
__host__ void deallocateMemory(T *d_a, T *d_b, T *d_c) {
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
    if (i + 1 >= argc) break;
    std::string option(argv[i]);
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

// Allocate host memory
int *h_a = (int *)malloc(M * K * sizeof(int));
int *h_b = (int *)malloc(K * N * sizeof(int));
int *h_c = (int *)malloc(M * N * sizeof(int));

// Initialize
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

// Allocate device memory (Sử dụng HIP_CHECK để tránh warning nodiscard)
int *d_a, *d_b, *d_c;
HIP_CHECK(hipMalloc((void **)&d_a, M * K * sizeof(int)));
HIP_CHECK(hipMalloc((void **)&d_b, K * N * sizeof(int)));
HIP_CHECK(hipMalloc((void **)&d_c, M * N* sizeof(int)));

copyFromHostToDevice<int>(h_a, h_b, d_a, d_b, M, N, K);

hipEvent_t start, stop;
float time;
HIP_CHECK(hipEventCreate(&start));
HIP_CHECK(hipEventCreate(&stop));

HIP_CHECK(hipEventRecord(start, 0));

executeKernel<int>(d_a, d_b, d_c, M, N, K);

HIP_CHECK(hipEventRecord(stop, 0));
HIP_CHECK(hipEventSynchronize(stop));

HIP_CHECK(hipEventElapsedTime(&time, start, stop));
printf("Time taken for GEMM: %f ms\n", time);
HIP_CHECK(hipEventDestroy(start));
HIP_CHECK(hipEventDestroy(stop));

std::cout << "Performance: " << 2LL*N*N*N/(time * 1e-3 * 1e9) << " GFLOP/s\n";

copyFromDeviceToHost<int>(d_c, h_c, M, N);
verifyResult<int>(h_a, h_b, h_c, M, N, K);
deallocateMemory<int>(d_a, d_b, d_c);
cleanUpDevice();

// Free host memory
free(h_a);
free(h_b);
free(h_c);

return 0;
}