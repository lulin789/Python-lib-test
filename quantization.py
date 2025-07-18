import numpy as np
np.set_printoptions(precision=5, suppress=True)

input_array = np.array([-7.43, -3.82, 0, 1.98, 10.12], dtype=np.float32)


# Symmetric quantization

scale_factor = (2**(8-1) - 1) / np.max(np.abs(input_array))

# quantization to int8
quantized_array = np.round(input_array * scale_factor).astype(np.int8)

# dequantization back to float32
dequantized_array = quantized_array.astype(np.float32) / scale_factor

print(f'FP32 input array: {input_array}')
print(f'Scale factor: {scale_factor}')
print(f'Int8 quantized array: {quantized_array}')
print(f'Dequantized FP32 array: {dequantized_array}')


# Asymmetric quantization

scale_factor = 2**8 / np.max(input_array - np.min(input_array))

zero_point = np.round(-scale_factor * np.min(input_array)) - 128

quantized_array  = np.round(input_array * scale_factor + zero_point).astype(np.int8)

dequantized_array = (quantized_array - zero_point) / scale_factor

print(f'Int8 quantized array: {quantized_array}')
print(f'Dequantized FP32 array: {dequantized_array}')






