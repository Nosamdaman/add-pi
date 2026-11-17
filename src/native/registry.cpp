#include <Python.h>
#include <torch/csrc/stable/library.h>

#include "add_pi.hpp"

// Define dummy module that can be loaded by CPython. This will load the library that torch grabs our implementations
// from.
#define CONCAT2(a, b) a ## b
#define CONCAT(a, b) CONCAT2(a, b)
#define DUMMY_MODULE_NAME CONCAT(PyInit_, NATIVE_MODULE_NAME)
#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)
#define NATIVE_MODULE_NAME_STR STR(NATIVE_MODULE_NAME)
PyMODINIT_FUNC DUMMY_MODULE_NAME(void) {
    static struct PyModuleDef module_def = {
        PyModuleDef_HEAD_INIT,
        NATIVE_MODULE_NAME_STR,
        NULL,
        -1,
        NULL,
    };
    return PyModule_Create(&module_def);
}

// Define the extension library and the CPU and CUDA implementations of it
STABLE_TORCH_LIBRARY(TORCH_EXTENSION_NAME, m) {
    m.def("add_pi(Tensor a) -> Tensor");
}
STABLE_TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CPU, m) {
    m.impl("add_pi", TORCH_BOX(&add_pi_cpu));
}
STABLE_TORCH_LIBRARY_IMPL(TORCH_EXTENSION_NAME, CUDA, m) {
    m.impl("add_pi", TORCH_BOX(&add_pi_cuda));
}
