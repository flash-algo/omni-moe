import triton
import triton.language as tl


@triton.jit
def silu(x):
    return x * tl.sigmoid(x)
