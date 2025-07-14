import torch

"""
This script using GPTQ quantized weights to perform matrix multiplication
and compare the results of fused and split matrix multiplication methods.
It also analyzes the precision loss between the two methods.
It requires the following files:
- out_bf16.pt: Reference output in bfloat16.
- x_u16.pt: Input tensor in bfloat16.
- w_u16.pt: Weights tensor in bfloat16.
- w_unpack.pt: Unpacked weights in 4-bit format.
- g_idx.pt: Group indices for weights.
- scale_grouped.pt: Scales for grouped weights.
- scale.pt: Scales for raw weights (not used in this script).
- zero_point_grouped.pt: Zero points for grouped weights.

It assumes the weights are quantized in a grouped manner.
It performs the following steps:
1. Load the data from the specified files.
2. Dequantize the weights using the provided scales and zero points.
3. Perform matrix multiplication using both fused and split methods.
4. Analyze the precision loss between the two methods.
5. Print the results of the precision loss analysis.

The results shows that the split matrix multiplication method cannot match the fused method,
probobly due to the floating-point didn't satisfied the multiplication and exchange law.
"""

# < ======= 1. Load data ======= >
data_path: str = '/home/lulinden/my_Tech/Python/numericbench/GPTQ'
out_ref = torch.load(f'{data_path}/out_bf16.pt')

x_bf16 = torch.load(f'{data_path}/x_u16.pt').view(torch.bfloat16)
w_bf16 = torch.load(f'{data_path}/w_u16.pt').view(torch.bfloat16)

w_u4 = torch.load(f'{data_path}/w_unpack.pt').reshape(w_bf16.shape[0], -1)

# scale_raw = torch.load(f'{data_path}/scale.pt').view(torch.bfloat16)
g_idx = torch.load(f'{data_path}/g_idx.pt')
scale_grouped = torch.load(f'{data_path}/scale_grouped.pt')

# zero_point_raw = torch.zeros_like(scale_raw, dtype=torch.uint8) + 8
zero_point_grouped = torch.zeros_like(scale_grouped, dtype=torch.uint8) + 8


# < ======= 2. Dequantize weights ======= >
dq_w = torch.zeros_like(w_u4, dtype=torch.bfloat16)

for group_idx in range(scale_grouped.shape[0]):
    # Get the scale for the current group
    scale = scale_grouped[group_idx, :]

    # Get the indices of the weights in this group
    weight_indices = torch.where(g_idx == group_idx)[0]

    # Dequantize the weights for this group
    dq_w[weight_indices] = (w_u4[weight_indices] -
                            zero_point_grouped[group_idx]) * scale

# Convert to float32 for computation

# < ======= 3. Method 1: Perform normal(fused) matrix multiplication ======= >
out_fused_bf16 = torch.matmul(x_bf16, dq_w)
out_fused_fp32 = torch.matmul(x_bf16.to(torch.float32), dq_w.to(torch.float32))


# < ======= 4. Method 2: Perform split matrix multiplication ======= >
all_group_out_bf16 = torch.zeros((112, 5, 512), dtype=torch.bfloat16)
all_group_out_fp32 = torch.zeros((112, 5, 512), dtype=torch.float32)

for group_idx in range(scale_grouped.shape[0]):
    x = x_bf16[:, 32 * group_idx:32 * (group_idx + 1)]

    single_group_out_bf16 = torch.matmul(
        x, dq_w[32 * group_idx:32 * (group_idx + 1), :])

    single_group_out_fp32 = torch.matmul(
        x.to(torch.float32), dq_w[32 * group_idx:32 * (group_idx + 1), :].to(torch.float32))

    all_group_out_bf16[group_idx, :, :] = single_group_out_bf16
    all_group_out_fp32[group_idx, :, :] = single_group_out_fp32

all_group_out_bf16 = all_group_out_bf16.sum(dim=0)
all_group_out_fp32 = all_group_out_fp32.sum(dim=0)


# < ======= 5. Precision Loss Analysis ======= >
print("=" * 60)
print("PRECISION LOSS ANALYSIS")
print("=" * 60)

# BFloat16 Analysis
print("\n🔍 BFloat16 Precision Analysis:")
print("-" * 40)
diff_bf16 = torch.abs(out_fused_bf16 - all_group_out_bf16)
max_diff_bf16 = torch.max(diff_bf16).item()
mean_diff_bf16 = torch.mean(diff_bf16).item()
rmse_bf16 = torch.sqrt(torch.mean(diff_bf16 ** 2)).item()


print(f"  Max absolute difference: {max_diff_bf16:.6e}")
print(f"  Mean absolute difference: {mean_diff_bf16:.6e}")
print(f"  RMSE: {rmse_bf16:.6e}")

# Float32 Analysis
print("\n🔍 Float32 Precision Analysis:")
print("-" * 40)
diff_fp32 = torch.abs(out_fused_fp32 - all_group_out_fp32)
max_diff_fp32 = torch.max(diff_fp32).item()
mean_diff_fp32 = torch.mean(diff_fp32).item()
rmse_fp32 = torch.sqrt(torch.mean(diff_fp32 ** 2)).item()

# Relative error (avoid division by zero)
rel_diff_fp32 = diff_fp32 / (torch.abs(out_fused_fp32) + 1e-8)
max_rel_diff_fp32 = torch.max(rel_diff_fp32).item()
mean_rel_diff_fp32 = torch.mean(rel_diff_fp32).item()

print(f"  Max absolute difference: {max_diff_fp32:.6e}")
print(f"  Mean absolute difference: {mean_diff_fp32:.6e}")
print(f"  RMSE: {rmse_fp32:.6e}")
