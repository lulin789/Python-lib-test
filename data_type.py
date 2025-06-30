import numpy as np
import tensorflow as tf
import time

bfloat16 = tf.bfloat16.as_numpy_dtype


def numerics_test():
    # This will work even without hardware BF16 support
    arr_bf16 = np.array([1.5, 2.7, 3.9], dtype=bfloat16)
    result = arr_bf16 * 2.0

    arr_fp16 = np.array([1.5, 2.7, 3.9], dtype=np.float16)
    result_fp16 = arr_fp16 * 2.0

    print(f"Array bf16: {arr_bf16}")
    print(f"Array fp16: {arr_fp16}")
    print('*'*30)
    print(f"Result: {result}")
    print(f"Result fp16: {result_fp16}")
    print('*'*30)
    print(f"Dtype: {result.dtype}")
    print(f"Dtype fp16: {result_fp16.dtype}")


def performance_test():
    size = 5000
    iterations = 1000

    # Create test arrays
    arr_fp32 = np.random.random(size).astype(np.float32)
    arr_bf16 = np.random.random(size).astype(bfloat16)
    arr_fp16 = np.random.random(size).astype(np.float16)  # Added FP16 array

    # Benchmark FP32
    start = time.time()
    for _ in range(iterations):
        result_fp32 = arr_fp32 * 2.0 + 1.0
    fp32_time = time.time() - start

    # Benchmark BF16 (software emulation)
    start = time.time()
    for _ in range(iterations):
        result_bf16 = arr_bf16 * 2.0 + 1.0
    bf16_time = time.time() - start

    # Benchmark FP16
    start = time.time()
    for _ in range(iterations):
        result_fp16 = arr_fp16 * 2.0 + 1.0
    fp16_time = time.time() - start

    print(f"FP32 time: {fp32_time:.4f}s")
    print(f"BF16 time: {bf16_time:.4f}s")
    print(f"FP16 time: {fp16_time:.4f}s")  # Added FP16 timing
    print(f"BF16 is {bf16_time/fp32_time:.2f}x slower")
    # Added FP16 comparison
    print(f"FP16 is {fp16_time/fp32_time:.2f}x slower")


def memory_test():
    # Memory usage comparison
    arr_fp32 = np.random.random(10000000).astype(np.float32)
    arr_bf16 = np.random.random(10000000).astype(bfloat16)

    print(f"FP32 memory: {arr_fp32.nbytes / 1024 / 1024:.2f} MB")
    print(f"BF16 memory: {arr_bf16.nbytes / 1024 / 1024:.2f} MB")
    print(f"Memory savings: {(1 - arr_bf16.nbytes/arr_fp32.nbytes)*100:.1f}%")

    # But computation is slower due to conversions
    def heavy_computation(arr):
        return np.sum(arr ** 2 + np.sqrt(np.abs(arr)))

    # Time the computations
    import time

    start = time.time()
    result_fp32 = heavy_computation(arr_fp32)
    fp32_compute_time = time.time() - start

    start = time.time()
    result_bf16 = heavy_computation(arr_bf16)
    bf16_compute_time = time.time() - start

    print(f"\nComputation times:")
    print(f"FP32: {fp32_compute_time:.4f}s")
    print(f"BF16: {bf16_compute_time:.4f}s")


if __name__ == "__main__":
    # numerics_test()

    # performance_test()

    memory_test()
