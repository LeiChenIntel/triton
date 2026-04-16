### Build Notes

Install with a virtualenv:

```shell
git clone https://github.com/triton-lang/triton.git
cd triton

python -m venv .venv --prompt triton
source .venv/bin/activate

pip install -r python/requirements.txt # build-time dependencies
export MAX_JOBS=18 # set up job number
pip install -e . # or use pip install -e . -v to print more logs
```

- Triton cache is under `/home/usr-name/.triton`.
- The default build path is under `./triton/python/build`.

CMake arguments to build project are dumped as,
```cmake
-G Ninja
-DCMAKE_EXPORT_COMPILE_COMMANDS=ON
-DLLVM_ENABLE_WERROR=ON
-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=./triton/python/triton/_C
-DTRITON_BUILD_TUTORIALS=OFF
-DTRITON_BUILD_PYTHON_MODULE=ON
-DPython3_EXECUTABLE:FILEPATH=./triton/.venv/bin/python3
-DPython3_INCLUDE_DIR=/usr/include/python3.10
-DTRITON_CODEGEN_BACKENDS=nvidia;amd
-DTRITON_WHEEL_DIR=./triton/python/triton
-DLLVM_EXTERNAL_LIT=./triton/.venv/bin/lit
-DLLVM_INCLUDE_DIRS=./.triton/llvm/llvm-f6ded0be-ubuntu-x64/include [under home/username]
-DLLVM_LIBRARY_DIR=./.triton/llvm/llvm-f6ded0be-ubuntu-x64/lib [under home/username]
-DLLVM_SYSPATH=./.triton/llvm/llvm-f6ded0be-ubuntu-x64 [under home/username]
-DCMAKE_BUILD_TYPE=TritonRelBuildWithAsserts
-DJSON_INCLUDE_DIR=./.triton/json//include
-DJSON_SYSPATH=./.triton/json/
-Dpybind11_INCLUDE_DIR='/tmp/pip-build-env-cmrmi3eq/overlay/lib/python3.10/site-packages/pybind11/include'
# set path to ./triton/.venv/lib/python3.10/site-packages/pybind11/include, do not use /tmp/..
-Dpybind11_DIR='/tmp/pip-build-env-cmrmi3eq/overlay/lib/python3.10/site-packages/pybind11/share/cmake/pybind11'
# set path to ./triton/.venv/lib/python3.10/site-packages/pybind11/share/cmake/pybind11, do not use /tmp/..
-DCUPTI_INCLUDE_DIR=./triton/third_party/nvidia/backend/include
-DROCTRACER_INCLUDE_DIR=./triton/third_party/amd/backend/include
```

By using these args, IDE can help to construct the project.

Need to rerun `pip install -e .` if changes in python code are not applied. Changes should be integrated into
virtual environment, and then they can take effect.

# Run test case
```
cd triton
pip install -e './python[tutorials]'
pip install torch
cd python/tutorials/
TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 python 01-vector-add.py > log
```
https://triton-lang.org/main/getting-started/tutorials/index.html

- Set `TRITON_ALWAYS_COMPILE=1` to compile kernel each time.
- Set `MLIR_ENABLE_DUMP=1` to dump IRs.

Issues:
```shell
UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.show()
```
Require to install UI `pip install PyQt5`
