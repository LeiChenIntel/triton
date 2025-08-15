import os
import importlib
import importlib.resources

import triton
import triton._C

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
