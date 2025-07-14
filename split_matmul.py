import torch


def test_split_matmul_int8():
    """
    Test the split matrix multiplication for int8 tensors.
    A large matrix multiplication is split into smaller groups.
    The results show that split matmul results match the direct computation.
    """
    # Create random int8 test case tensors
    torch.manual_seed(42)  # For reproducible results
    A = torch.randint(-128, 127, (16, 256), dtype=torch.int8)
    B = torch.randint(-128, 127, (256, 32), dtype=torch.int8)

    # Perform matrix multiplication (convert to int32 to avoid overflow)
    # Method 1: Direct computation
    C = torch.matmul(A.to(torch.int32), B.to(torch.int32))

    # Method 2: Grouped computation
    all_group_out = torch.zeros((16, 16, 32), dtype=torch.int32)
    for group_idx in range(16):
        x = A[:, 16 * group_idx: 16 * (group_idx + 1)]

        # Convert to int32 to avoid overflow in partial computations
        single_group_out = torch.matmul(
            x.to(torch.int32), B[16 * group_idx:16 * (group_idx + 1), :].to(torch.int32))

        all_group_out[group_idx, :] = single_group_out
    # Sum all group outputs along dimension 0
    all_group_out = all_group_out.sum(dim=0)

    assert torch.equal(C, all_group_out)


if __name__ == "__main__":
    test_split_matmul_int8()
    print("Split matmul test passed!")
