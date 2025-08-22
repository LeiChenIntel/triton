"""
This is an example to enable a new backend in Triton
"""

import torch
import triton
import triton.language as tl


@triton.jit
def copy_kernel(x_ptr, output_ptr):
    x = tl.load(x_ptr, mask=True)
    tl.store(output_ptr, x, mask=True)


def copy(x: torch.Tensor, output: torch.Tensor):
    if output is None:
        output = torch.empty_like(x)
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    copy_kernel[grid](x, output)
    return output


torch.manual_seed(0)
size = 64

triton.runtime.driver.set_active_to_xpu()
x = torch.rand(size, device='cpu')
output = copy(x, None)
