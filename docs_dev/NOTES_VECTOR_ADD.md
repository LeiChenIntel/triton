
With size=1024 and BLOCK_SIZE=1024, there is only 1 program (pid=0). 
But tl.device_print prints once per SIMD lane group (sub-group/warp), not once per program.

Triton's compiler chose a vectorization factor of 8 (8 × float32 = 256-bit vector width).
Instead of generating 1024 scalar operations per warp/thread, it packs 8 scalar fadds into each thread's instruction stream.
Then it can dump 1024/8=128 logs when executing the kernel.
After the pass `ConvertTritonGPUToLLVM`, 8 floats are added per kernel and tid is set to 128 in the LLVM IR.
```python
pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
tl.device_print("pid: ", pid) # print 128 times since there are 128 threads
...
torch.manual_seed(0)
size = 1024 # reset the size and comment @triton.testing.perf_report to avoid massive logs.
```
1024 offsets are generated if the device print is inserted into the kernel,
```python
offsets = block_start + tl.arange(0, BLOCK_SIZE)
tl.device_print("offsets: ", offsets)
```
This is because 8 `%60 = llvm.call @vprintf(%1, %54) : (!llvm.ptr, !llvm.ptr) -> i32 loc(#loc1)` are
inserted in the LLVM IR in the `llvm.func @add_kernel`, and there are 128 threads, so 128×8=1024 logs are generated.

Thread Id (tid) can be printed in the kernel to confirm that 128 threads are launched:
```python
# tl.device_print("pid: ", pid)
# Get hardware thread ID via inline PTX
tid = tl.inline_asm_elementwise(
    "mov.u32 $0, %tid.x;",
    "=r",
    [],
    dtype=tl.int32,
    is_pure=True,
    pack=1
)
tl.device_print("tid: ", tid)
```
