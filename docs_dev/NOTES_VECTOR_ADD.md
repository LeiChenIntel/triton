
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
size = 1024
```
