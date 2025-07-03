import torch

# Set random seed for reproducibility
torch.manual_seed(42)


def matmul_to_conv2d():
    """
    This function demonstrates how matrix multiplication (matmul) can be converted 
    to 2D convolution (Conv2D) operations through tensor reshaping.

    Mathematical equivalence:
    - Matmul: Y = X @ W, where X is (M, N) and W is (N, K) -> Y is (M, K)
    - Conv2D: Y = conv2d(X_reshaped, W_reshaped) with proper reshaping

    Key insight: Conv2D with 1x1 kernels and appropriate tensor reshaping 
    can perform the same computation as matrix multiplication.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Create example tensors to demonstrate matmul -> Conv2D conversion
    # Original shapes: input_raw (8, 3), weight_raw (3, 5)
    input_raw = torch.randint(0, 10, (8, 3))
    weight_raw = torch.randint(0, 10, (3, 5))
    output_matmul = torch.matmul(input_raw, weight_raw)

    # CONVERSION METHOD 1: Reshape to mimic 1D convolution
    # =====================================================
    #
    # For matmul: Y = input_raw @ weight_raw
    # We reshape tensors to use Conv2D:
    #
    # Input tensor transformation:
    # - Original: (8, 3) -> represents 8 sequence positions, 3 input features
    # - Reshaped: (1, 3, 8, 1) -> (batch_size=1, channels=3, height=8, width=1)
    #
    # Weight tensor transformation:
    # - Original: (3, 5) -> 3 input features, 5 output features
    # - Reshaped: (5, 3, 1, 1) -> (out_channels=5, in_channels=3, kernel_h=1, kernel_w=1)
    #
    # Conv2D with 1x1 kernel performs: output[i,j] = sum(w[i,k] * input[k,j]) for each position j
    # This is exactly the same as matrix multiplication!

    # Create input tensor: (batch_size, channels, height, width) / (N, C_in, H, W)
    input_tensor = input_raw.transpose(0, 1).reshape(1, 3, 8, 1)
    # Create weight tensor: (out_channels, in_channels, kernel_height, kernel_width) / (C_out, C_in, k_h, k_w)
    weight = weight_raw.transpose(0, 1).reshape(5, 3, 1, 1)

    # CONVERSION METHOD 2: Alternative reshaping for better memory layout
    # ==================================================================
    #
    # Instead of using (1, 3, 8, 1), we can reshape to (1, 3, 4, 2)
    # This creates a more balanced 2D layout that might be more efficient
    # for certain hardware accelerators or when the sequence is very long.
    #
    # The convolution still performs the same computation because:
    # - 1x1 kernel means no spatial mixing between height/width dimensions
    # - The transpose and reshape preserve the mathematical relationships
    # - Final output can be reshaped back to match the matmul result

    # For better memory layout, we can reshape to a more balanced 2D tensor
    input_tensor_opt = input_raw.transpose(0, 1).reshape(1, 3, 4, 2)
    weight_opt = weight_raw.transpose(0, 1).reshape(5, 3, 1, 1)

    # PERFORM CONVOLUTIONS
    # ====================
    # Both convolutions should produce identical results to the original matmul
    # because they're performing the same mathematical operation with different tensor layouts

    # Perform 2D convolution - Method 1 (8x1 layout)
    output = torch.conv2d(input_tensor, weight, stride=1, padding=0)

    # Perform 2D convolution - Method 2 (4x2 layout)
    output_opt = torch.conv2d(
        input_tensor_opt, weight_opt, stride=1, padding=0)

    # VERIFICATION 1: Both Conv2D methods should produce the same result
    # =================================================================
    # Even though the input tensors have different spatial dimensions (8x1 vs 4x2),
    # the 1x1 convolution kernel ensures the same computation is performed
    assert (output_opt.reshape(5, 8) == output.reshape(5, 8)).all()
    print("✓ Both Conv2D methods produce identical results")

    # VERIFICATION 2: Conv2D result should match matrix multiplication
    # ===============================================================
    # This is the key validation: proving that Conv2D can replace matmul
    assert (output_matmul.transpose(0, 1) == output.reshape(5, 8)).all()
    print("✓ Matrix multiplication and Conv2D produce identical results!")

    # MATHEMATICAL EXPLANATION:
    # ========================
    # Matmul: Y[i,j] = Σ(k=0 to 2) X[i,k] * W[k,j]  (input @ weight)
    # Conv2D: Y[j,h,w] = Σ(k=0 to 2) W[j,k,0,0] * X[k,h,w] (with 1x1 kernel)
    #
    # When we reshape and transpose the Conv2D output back to 2D, we get the same result as matmul!

    print("\n" + "="*60)
    print("CONVERSION SUMMARY:")
    print("="*60)
    print(
        f"Original matmul: ({input_raw.shape}) @ ({weight_raw.shape}) = ({output_matmul.shape})")
    print(
        f"Conv2D method 1: conv2d({input_tensor.shape}, {weight.shape}) = {output.shape}")
    print(
        f"Conv2D method 2: conv2d({input_tensor_opt.shape}, {weight_opt.shape}) = {output_opt.shape}")
    print("All three methods produce mathematically equivalent results!")
    print("="*60)


if __name__ == '__main__':
    matmul_to_conv2d()
