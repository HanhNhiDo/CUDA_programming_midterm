#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>
#include <iostream>
#include <tuple>
#include <string>
#include <cmath>
const int WARPSIZE = 64;

using namespace std;

// Macro kiểm tra lỗi HIP
#define HIP_CHECK(command) { \
hipError_t status = command; \
if (status != hipSuccess) { \
    fprintf(stderr, "Error: HIP reports %s at line %d\n", hipGetErrorString(status), __LINE__); \
    std::exit(EXIT_FAILURE); \
} \
}

template<typename T>
__host__ void verifyResult(T *h_a, T *h_b, T *h_c, int M, int N, int K) {
// Check một số mẫu ngẫu nhiên hoặc check toàn bộ
bool correct = true;
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
    T sum = 0;
    for (int k = 0; k < K; k++) {
        sum += h_a[i * K + k] * h_b[k * N + j];
    }
    // So sánh float với sai số nhỏ
    if (abs(h_c[i * N + j] - sum) > 1e-2) {
        // In lỗi đầu tiên tìm thấy để debug
        // printf("Mismatch at [%d][%d]: Expected %f, Got %f\n", i, j, sum, h_c[i * N + j]);
        correct = false;
        // break; 
    }
    }
}
if (correct) printf("Result is Correct!\n");
else printf("Result is Incorrect!\n");
}

template <const int BM, const int BN, const int BK, const int rowStrideA,
        const int rowStrideB>
__device__ void loadFromGmem(int N, int K, const float *A, const float *B,
                            float *As, float *Bs, int innerRowA, int innerColA,
                            int innerRowB, int innerColB) {
// Load A vào Shared Memory (có chuyển vị để tối ưu truy cập)
for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
    const float4 tmp = reinterpret_cast<const float4 *>(
        &A[(innerRowA + offset) * K + innerColA * 4])[0];
    As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
    As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
    As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
    As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
}

// Load B vào Shared Memory
for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
    reinterpret_cast<float4 *>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
        reinterpret_cast<const float4 *>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
}
}

template <const int BM, const int BN, const int BK, const int WM, const int WN,
        const int WMITER, const int WNITER, const int WSUBM, const int WSUBN,
        const int TM, const int TN>
__device__ void
processFromSmem(float *regM, float *regN, float *threadResults, const float *As,
                const float *Bs, const uint warpRow, const uint warpCol,
                const uint threadRowInWarp, const uint threadColInWarp) {
for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
    // 1. Load dữ liệu từ Shared Mem vào Registers
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint i = 0; i < TM; ++i) {
        regM[wSubRowIdx * TM + i] =
            As[(warpRow * WM + wSubRowIdx * WSUBM +
                threadRowInWarp * TM + i) + dotIdx * BM];
    }
    }
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
    for (uint i = 0; i < TN; ++i) {
        regN[wSubColIdx * TN + i] =
            Bs[(dotIdx * BN) + warpCol * WN + wSubColIdx * WSUBN +
            threadColInWarp * TN + i];
    }
    }

    // 2. Tính toán Outer Product (Math heavy part)
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
        for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        (wSubColIdx * TN) + resIdxN] +=
                regM[wSubRowIdx * TM + resIdxM] *
                regN[wSubColIdx * TN + resIdxN];
        }
        }
    }
    }
}
}

template<typename T, size_t BM, size_t BN, size_t BK, size_t TM, size_t TN, size_t WM, size_t WN, size_t WNITER>
__global__ void gemm_kernel(T* A, T* B, T* C, size_t M, size_t N, size_t K) {
// Với Wave64, nếu block size là 128 thì có đúng 2 Wavefronts.
const uint NUM_THREADS = 128; 
const uint cRow = blockIdx.y;
const uint cCol = blockIdx.x;

// Xác định vị trí của Wavefront (Warp)
const uint warpIdx = threadIdx.x / WARPSIZE; 
const uint warpCol = warpIdx % (BN / WN);
const uint warpRow = warpIdx / (BN / WN);

// Tính toán kích thước Sub-tile cho Wave64
// Logic này đảm bảo WMITER >= 1. Nếu = 0 sẽ lỗi.
constexpr uint WMITER = (WM * WN) / (WARPSIZE * TM * TN * WNITER);
constexpr uint WSUBM = WM / WMITER; 
constexpr uint WSUBN = WN / WNITER; 

// Vị trí của thread trong Wavefront
const uint threadIdxInWarp = threadIdx.x % WARPSIZE;
const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];

A += cRow * BM * K;
B += cCol * BN;
C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

// Tính toán chỉ số load Global Memory
// Vectorized load float4 (128 bit)
const uint innerRowA = threadIdx.x / (BK / 4);
const uint innerColA = threadIdx.x % (BK / 4);
constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;
const uint innerRowB = threadIdx.x / (BN / 4);
const uint innerColB = threadIdx.x % (BN / 4);
constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

float threadResults[WMITER * TM * WNITER * TN] = {0.0};
float regM[WMITER * TM] = {0.0};
float regN[WNITER * TN] = {0.0};

// Main Loop
for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    loadFromGmem<BM, BN, BK, rowStrideA, rowStrideB>(
        N, K, A, B, As, Bs, innerRowA, innerColA, innerRowB, innerColB);
    __syncthreads();
    
    processFromSmem<BM, BN, BK, WM, WN, WMITER, WNITER, WSUBM, WSUBN, TM,
                        TN>(regM, regN, threadResults, As, Bs, warpRow, warpCol,
                            threadRowInWarp, threadColInWarp);
    A += BK;    
    B += BK * N; 
    __syncthreads();
}

// Ghi kết quả ra Global Memory
for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
    for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
    float *C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
    for (uint resIdxM = 0; resIdxM < TM; resIdxM += 1) {
        for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
        float4 tmp = reinterpret_cast<float4 *>(
            &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0];
        
        const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                        wSubColIdx * TN + resIdxN;
        tmp.x = threadResults[i + 0];
        tmp.y = threadResults[i + 1];
        tmp.z = threadResults[i + 2];
        tmp.w = threadResults[i + 3];
        
        reinterpret_cast<float4 *>(
            &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                        threadColInWarp * TN + resIdxN])[0] = tmp;
        }
    }
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

template<typename T, const uint BM, const uint BN, const uint BK, const uint TM, const uint TN, const uint WM, const uint WN, const uint WNITER>
__host__ void executeKernel(T* d_a, T* d_b, T* d_c, size_t M, size_t N, size_t K) {
// Với Wave64, WM=128, WN=64, BM=128, BN=128
// Số Waves M = BM/WM = 128/128 = 1
// Số Waves N = BN/WN = 128/64 = 2
// Tổng Waves = 2.
// Tổng Threads = 2 * 64 = 128 threads.
dim3 block(128); 
dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);

gemm_kernel<T, BM, BN, BK, TM, TN, WM, WN, WNITER><<<grid, block>>>(d_a, d_b, d_c, M, N, K);

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

// CẤU HÌNH TILING CHO AMD :
// BM=128, BN=128, BK=16, TM=8, TN=4
// WM=128 , WN=64, WNITER=4
executeKernel<float, 128, 128, 16, 8, 4, 128, 64, 4>(d_a, d_b, d_c, M, N, K);

HIP_CHECK(hipEventRecord( stop, 0 ));
HIP_CHECK(hipEventSynchronize( stop ));

HIP_CHECK(hipEventElapsedTime( &time, start, stop ));
printf("Time taken for GEMM: %f ms\n", time);

HIP_CHECK(hipEventDestroy( start ));
HIP_CHECK(hipEventDestroy( stop ));

std::cout << "Performance: " << 2LL*M*N*K/(time * 1e-3 * 1e9) << " GFLOP/s\n";

copyFromDeviceToHost<float>(d_c, h_c, M, N);
verifyResult<float>(h_a, h_b, h_c, M, N, K);
deallocateMemory<float>(d_a, d_b, d_c);
cleanUpDevice();

free(h_a);
free(h_b);
free(h_c);

return 0;
}