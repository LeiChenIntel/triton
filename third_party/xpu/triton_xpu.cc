#include <pybind11/stl.h>

namespace py = pybind11;

void init_triton_xpu(py::module &&m) {
  m.doc() = "Python bindings to the XPU Triton backend";
}
