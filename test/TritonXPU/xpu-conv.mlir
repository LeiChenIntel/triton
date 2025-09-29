// RUN: TRITON_ALWAYS_COMPILE=1 python %python_test_path/00-conv-xpu.py | FileCheck %s

// CHECK: Target: xpu
// CHECK: tt.func public @conv_kernel
// CHECK: %c0_i32 = arith.constant 0 : i32
// CHECK: %0 = tt.addptr %arg0, %c0_i32 : !tt.ptr<f32>, i32
// CHECK: %1 = tt.splat %0 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHEKC: %2 = tt.load %1 : tensor<1x!tt.ptr<f32>>
// CHECK: %3 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f32>, i32
// CHECK: %4 = tt.splat %3 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHECK: %5 = tt.load %4 : tensor<1x!tt.ptr<f32>>
// CHECK: %6 = tt.conv(%2, %5) : tensor<1xf32>, tensor<1xf32> -> tensor<1xf32>
// CHECK: %7 = tt.addptr %arg2, %c0_i32 : !tt.ptr<f32>, i32
// CHECK: %8 = tt.splat %7 : !tt.ptr<f32> -> tensor<1x!tt.ptr<f32>>
// CHECK: tt.store %8, %6 : tensor<1x!tt.ptr<f32>>
// CHECK: tt.return
