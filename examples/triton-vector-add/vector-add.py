import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # *Pointers* to matrices
    y_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,  # Compile-time constant
):
    # Create program ID for this thread
    pid = tl.program_id(axis=0)

    # Calculate start index for this thread block
    block_start = pid * BLOCK_SIZE

    # Create offsets for this thread block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Boundary check
    mask = offsets < n_elements

    # Load data using pointer arithmetic
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # Simple computation
    output = x + y

    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def add_vectors(x: torch.Tensor, y: torch.Tensor):
    assert x.shape == y.shape
    assert x.is_cuda and y.is_cuda

    n_elements = x.numel()
    output = torch.empty_like(x)

    # Grid configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch kernel
    add_kernel[grid](
        x, y, output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output

# Test it
def main():
    size = 98432  # Deliberately non-power-of-2

    # Create input tensors
    x = torch.ones(size, device='cuda')
    y = torch.ones(size, device='cuda') * 2

    # Run kernel
    output = add_vectors(x, y)

    # Verify
    print(f"First element: {output[0]}")  # Should be 3
    print(f"Max error: {torch.max(torch.abs(output - (x + y)))}")  # Should be 0

if __name__ == '__main__':
    main()
