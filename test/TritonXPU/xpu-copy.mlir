// RUN: TRITON_ALWAYS_COMPILE=1 python %python_test_path/00-copy-xpu.py | FileCheck %s

// CHECK: Target: xpu
// CHECK: tt.func public @copy_kernel
// CHECK: %0 = tt.load %arg0, %true : !tt.ptr<f32>
// CHECK: tt.store %arg1, %0, %true : !tt.ptr<f32>
// CHECK: tt.return
