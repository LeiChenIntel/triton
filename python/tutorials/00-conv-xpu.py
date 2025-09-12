"""
This is an example to enable a new backend in Triton
"""

import torch
import triton
import triton.language as tl


@triton.jit
def conv_kernel(x_ptr, w_ptr, output_ptr):
    x = tl.load(x_ptr, mask=True)
    w = tl.load(w_ptr, mask=True)
    y = tl.conv(x, w)
    tl.store(output_ptr, y, mask=True)


def conv(xval: torch.Tensor, wval: torch.Tensor, yval: torch.Tensor):

    if yval is None:
        yval = torch.empty_like(torch.randn(1, 16, 1, 1, device='cpu'))

    # Redefine the grid
    grid = lambda meta: (11,)
    print(xval.dtype)
    print(wval.dtype)
    print(yval.dtype)
    conv_kernel[grid](xval, wval, yval)
    return yval


torch.manual_seed(0)
size = 64

triton.runtime.driver.set_active_to_xpu()
x = torch.randn(1, 32, 1, 1, device='cpu')
w = torch.randn(16, 32, 1, 1, device='cpu')
output = conv(x, w, None)
