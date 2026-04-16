### LLVM Notes

LIT CHECK labels definitions:
* CHECK: 
  - Matches lines in order — each CHECK must appear after the previous one in the output
  - Strict sequential matching
```text
// CHECK: %[[CST:.*]] = arith.constant dense<0.000000e+00> : tensor<128x32xf16>
// CHECK: %[[EXTSI0:.*]] = arith.extsi ...   ← must appear AFTER CST
```

* CHECK-DAG:
    - Matches lines in any order within a DAG group (consecutive CHECK-DAG lines)
    - Useful for constants/definitions that may be reordered by the compiler (e.g., CSE, canonicalization)
    - A CHECK-DAG group ends when a non-CHECK-DAG directive is encountered
```text
// CHECK-DAG: %[[C0_I32:.*]] = arith.constant 0 : i32
// CHECK-DAG: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK-DAG: %[[C128_I64:.*]] = arith.constant 128 : i64  ← these 3 can appear in any order
// CHECK: %[[CST:.*]] = ...   ← this must appear AFTER all the DAG matches above
```

* CHECK-SAME:
    - Matches content on the same line as the previous CHECK/CHECK-SAME
    - Used to verify multiple patterns on a single output line, like function signatures
```text
// CHECK-LABEL: tt.func public @rewrite_for(
// CHECK-SAME:     %[[ARG0:[a-zA-Z0-9_]+]]: !tt.ptr<f16>   ← same line as above
// CHECK-SAME:     %[[ARG1:[a-zA-Z0-9_]+]]: !tt.ptr<f16>   ← still same line
```
| Directive    | Order                          | Line Behavior                        |
|--------------|--------------------------------|--------------------------------------|
| `CHECK`      | Sequential                     | New line                             |
| `CHECK-DAG`  | Any order (within group)       | New line                             |
| `CHECK-SAME` | N/A (continues previous match) | Same line as previous CHECK/CHECK-SAME |


### Triton Notes

The relation between grid and pid

Grid defines the total number of parallel programs
The grid parameter specifies how many program instances will run in parallel:
```python
grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
```
For example, if you have 1000 elements and BLOCK_SIZE=256:
* triton.cdiv(1000, 256) = ceil(1000/256) = 4
* So grid = (4,) means 4 programs will run in parallel

PID identifies which specific program instance is running
Each program instance gets a unique identifier from 0 to grid_size-1:
```python
pid = tl.program_id(axis=0)  # Returns 0, 1, 2, or 3 in the above example
```

How they work together for data partitioning
Each program uses its PID to determine which data chunk to process:
```python
block_start = pid * BLOCK_SIZE
offsets = block_start + tl.arange(0, BLOCK_SIZE)
```
Concrete example with 1000 elements, BLOCK_SIZE=256:
* Grid size: 4 programs
* Program 0 (pid=0): processes elements 0-255 (0×256 to 0×256+255)
* Program 1 (pid=1): processes elements 256-511 (1×256 to 1×256+255)
* Program 2 (pid=2): processes elements 512-767 (2×256 to 2×256+255)
* Program 3 (pid=3): processes elements 768-999 (3×256 to 3×256+255, masked at 1000)
* offsets is a vector of indices that each program uses to access memory locations
  * tl.arange(0, BLOCK_SIZE) creates a vector [0, 1, 2, ..., BLOCK_SIZE-1]
  * block_start is the starting index for this program's data chunk
  * offsets becomes [block_start, block_start+1, block_start+2, ..., block_start+BLOCK_SIZE-1]

Example:
If BLOCK_SIZE=4 and pid=2:
* block_start = 2 * 4 = 8
* tl.arange(0, 4) = [0, 1, 2, 3]
* offsets = 8 + [0, 1, 2, 3] = [8, 9, 10, 11]
  
Usage in memory operations:
```python
x = tl.load(x_ptr + offsets, mask=mask)
```
This loads elements from memory addresses:
```
x_ptr + 8 (element at index 8)
x_ptr + 9 (element at index 9)
x_ptr + 10 (element at index 10)
x_ptr + 11 (element at index 11)
```

Visual representation
```
Data:     [0][1][2]...[255][256][257]...[511][512]...[767][768]...[999]
          |----pid=0----|----pid=1----|----pid=2----|----pid=3----|
Grid:     (4,) - means 4 parallel programs
```

Key relationship
* Grid = "How many parallel workers do we need?"
* PID = "Which worker am I, and what data should I process?"

The grid size is calculated to ensure all data gets processed by dividing the total work by the work each program does (BLOCK_SIZE),
while PID ensures each program knows exactly which portion of the data is its responsibility.

Example and Understanding for 01-vector-add.py
Let elements = 2000, BLOCK_SIZE = 1024, pid = 1
grid can be (2,) - means 2 parallel programs, pid can be 0 or 1
```mlir
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/leichen1/develop/triton-origin/triton/python/tutorials/01-vector-add.py":28:0), %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/leichen1/develop/triton-origin/triton/python/tutorials/01-vector-add.py":28:0), %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} loc("/home/leichen1/develop/triton-origin/triton/python/tutorials/01-vector-add.py":28:0), %arg3: i32 {tt.divisibility = 16 : i32} loc("/home/leichen1/develop/triton-origin/triton/python/tutorials/01-vector-add.py":28:0)) attributes {noinline = false} {
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    // get pid
    %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc3)
    // block_start = pid * BLOCK_SIZE, pointer start address
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc4)
    // tl.arange(0, BLOCK_SIZE)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32> loc(#loc5)
    // block_start -> [block_start, block_start, ..., block_start]
    %4 = arith.addi %3, %2 : tensor<1024xi32> loc(#loc5)
    // get offsets: offsets = block_start + tl.arange(0, BLOCK_SIZE)
    // [block_start + 0, block_start + 1, ..., block_start + (BLOCK_SIZE-1)]
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32> loc(#loc6)
    // n_elements -> [n_elements, n_elements, ..., n_elements]
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32> loc(#loc6)
    // get mask: mask = offsets < n_elements
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc7)
    // x_ptr -> [x_ptr, x_ptr, ..., x_ptr]
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc7)
    // x_ptr + offsets, get the actual memory addresses to load in this program
    %9 = tt.load %8, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc8)
    // load(x_ptr + offsets, mask=mask), load with mask
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc9)
    // y_ptr -> [y_ptr, y_ptr, ..., y_ptr]
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc9)
    // y_ptr + offsets, get the actual memory addresses to load in this program
    %12 = tt.load %11, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc10)
    // load(y_ptr + offsets, mask=mask), load with mask
    %13 = arith.addf %9, %12 : tensor<1024xf32> loc(#loc11)
    // x + y
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc12)
    // out_ptr -> [out_ptr, out_ptr, ..., out_ptr]
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc12)
    // out_ptr + offsets, get the actual memory addresses to store in this program
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc13)
    // store(out_ptr + offsets, output, mask=mask), store with mask
    tt.return loc(#loc14)
  } loc(#loc)
} loc(#loc)
```
