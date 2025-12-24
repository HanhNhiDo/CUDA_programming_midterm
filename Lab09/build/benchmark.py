import subprocess
import re
import os
import statistics
import time

# --- CẤU HÌNH ---

# 1. Danh sách file thực thi (Phiên bản O0 như bạn yêu cầu)
executables = [
    #"./naive",
    #"./tiled",
    #"./tiled_1d",
    #"./tiled_2d",
    #"./vectorize",
    #"./warp_tiled"
]

# 2. Các bộ dữ liệu MxNxK cần test
configurations = [
    (1024, 1024, 128),
    (1024, 1024, 1024),
    (512, 2048, 4096),
    (8192, 8192, 8192)  # Case này rất nặng với O0
]

# --- HÀM HỖ TRỢ ---

def get_runs_and_timeout(M, N, K):
    """
    Điều chỉnh số lần chạy và thời gian chờ dựa trên kích thước ma trận.
    Với -O0, case 8192 chạy rất chậm nên cần giảm số lần test.
    """
    total_ops = M * N * K
    if total_ops >= 8192 * 8192 * 8192:
        return 3, 300  # Chạy 3 lần, timeout 300s (5 phút) cho case siêu lớn
    elif total_ops >= 512 * 2048 * 4096:
        return 5, 60   # Chạy 5 lần, timeout 60s
    else:
        return 10, 30  # Chạy 10 lần cho các case nhỏ

def parse_performance(output):
    match = re.search(r"Performance:\s*([\d\.]+)\s*GFLOP/s", output)
    return float(match.group(1)) if match else None

def parse_time(output):
    match = re.search(r"Time taken for GEMM:\s*([\d\.]+)\s*ms", output)
    return float(match.group(1)) if match else None

def main():
    # In tiêu đề bảng
    print(f"{'Config':<18} | {'Executable':<15} | {'Runs':<4} | {'Avg Time (ms)':<15} | {'Avg Perf (GFLOP/s)':<20}")
    print("=" * 85)

    for M, N, K in configurations:
        config_str = f"{M}x{N}x{K}"
        runs, timeout_sec = get_runs_and_timeout(M, N, K)
        
        # In dòng phân cách cho mỗi cấu hình mới để dễ nhìn
        print(f"--- Testing Config: {config_str} ---")

        for exe in executables:
            if not os.path.exists(exe):
                print(f"{config_str:<18} | {exe:<15} | MISS | File not found")
                continue

            perfs = []
            times = []
            
            # Tạo lệnh chạy: ./naiveo0 -m 1024 -n 1024 -k 128
            cmd = [exe, "-m", str(M), "-n", str(N), "-k", str(K)]

            print(f"{config_str:<18} | {exe:<15} | {runs:<4} | ", end="", flush=True)

            for i in range(runs):
                try:
                    # Run với timeout để tránh treo máy ở case lớn
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
                    
                    if result.returncode != 0:
                        print("x", end="", flush=True) # Lỗi Runtime
                        continue

                    output = result.stdout
                    perf = parse_performance(output)
                    time_val = parse_time(output)

                    if perf and time_val:
                        perfs.append(perf)
                        times.append(time_val)
                        print(".", end="", flush=True)
                    else:
                        print("?", end="", flush=True) # Không parse được

                except subprocess.TimeoutExpired:
                    print("T", end="", flush=True) # Quá thời gian
                except Exception as e:
                    print("E", end="", flush=True) # Lỗi khác

            # Tính toán và in kết quả trung bình
            if perfs:
                avg_perf = statistics.mean(perfs)
                avg_time = statistics.mean(times)
                print(f" -> {avg_time:.2f} ms | {avg_perf:.4f} GFLOP/s")
            else:
                print(" -> No Data")

        print("-" * 85)

if __name__ == "__main__":
    main()