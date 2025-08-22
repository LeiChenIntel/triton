import os
import importlib
import importlib.resources

import triton
import triton._C
from triton.backends.driver import DriverBase
from triton.backends.compiler import GPUTarget

print("xpu backend driver")

_dirname = os.getenv("TRITON_SYS_PATH", default="/usr/local")
# for locating libTritonCPURuntime
try:
    _triton_C_dir = importlib.resources.files(triton).joinpath("_C")
except AttributeError:
    # resources.files() doesn't exist for Python < 3.9
    _triton_C_dir = importlib.resources.path(triton, "_C").__enter__()

include_dirs = [os.path.join(_dirname, "include")]
library_dirs = [os.path.join(_dirname, "lib"), _triton_C_dir]
libraries = ["stdc++"]


class XPUDriver(DriverBase):
    # Need to define abstract methods in DriverBase
    def __init__(self):
        super().__init__()

    @staticmethod
    def is_active():
        return True

    def get_current_device(self):
        # Required by runtime/jit.py
        return 0

    def get_current_stream(self, device):
        # Required by runtime/jit.py
        return 0

    def get_current_target(self):
        cpu_arch = 1234
        return GPUTarget("xpu", cpu_arch, 0)

    def get_benchmarker(self):
        from triton.testing import do_bench
        return do_bench
