project(
    'add_pi',
    ['cpp', 'cuda'],
    default_options: ['cpp_std=c++20',
    'python.install_env=auto'],
    meson_version: '>= 1.3',
    version: '0.1.0',
)

# Define all the source code
source_directory = 'src'
native_source_directory = source_directory / 'native'
native_source_files = [
    native_source_directory / 'add_pi.cpp',
    native_source_directory / 'add_pi.cu',
    native_source_directory / 'registry.cpp',
]
python_source_directory = source_directory / 'python'
python_source_files = [ python_source_directory / '__init__.py' ]

# Get the name of the directory we'll install the package into, this will be the project name
package_directory = meson.project_name()

# Get the python installation we'll install into. Note that we use pure: false here as we want both the native and non-
# native code to be installed into the same platform native location.
py = import('python').find_installation(modules: ['torch'], pure: false)


# Define our stable ABI targets
python_stable_abi_target = '3.12'
pytorch_stable_abi_target = '2.13'
pytorch_stable_abi_target_major = pytorch_stable_abi_target.split('.')[0]
pytorch_stable_abi_target_minor = pytorch_stable_abi_target.split('.')[1]
stable_abi_script_template = 'print(hex(((0 + @0@) << 56) | ((0 + @1@) << 48)))'
stable_abi_script = stable_abi_script_template.format(pytorch_stable_abi_target_major, pytorch_stable_abi_target_minor)
stable_abi_define = run_command(py, '-c', stable_abi_script, check: true).stdout().strip()

# Define the CUDA version and architecture we're targeting
cuda_target = '13.2'
cuda_arch_list = '7.5;8.6;8.7;8.9;9.0;10.0;12.0+PTX'

# These are the modules we require from the CUDA Toolkit. For now, all we need is the main cuda runtime.
cuda_modules = ['cudart']

# These are paths and module names related to the PyTorch dependency. Note that they are relative to wherever the
# PyTorch package is installed, we'll fill in the actual location later.
torch_package_base = 'torch'
base_torch_include_paths = [
    torch_package_base / 'include',
    torch_package_base / 'include' / 'torch' / 'csrc' / 'api' / 'include',
]
base_torch_library_path = torch_package_base / 'lib'
torch_library_names = ['torch', 'c10', 'c10_cuda', 'torch_cpu', 'torch_cuda']

# This will be the name of the compiled shared library
native_so_module = '_native'

# These are the compilers we support
cpp_compilers = ['gcc', 'msvc']
cuda_compilers = ['nvcc']

# Define the common macro defines
common_defines = [
    '-DTORCH_EXTENSION_NAME=' + meson.project_name(),
    '-DNATIVE_MODULE_NAME=' + native_so_module,
    '-DTORCH_TARGET_VERSION=' + stable_abi_define,
]

# Define the common C++ compiler arguments
cpp_args = []

# Define the common CUDA compiler arguments
cuda_args = [
    '-DUSE_CUDA',
    '-D__CUDA_NO_HALF_OPERATORS__',
    '-D__CUDA_NO_HALF_CONVERSIONS__',
    '-D__CUDA_NO_BFLOAT16_CONVERSIONS__',
    '-D__CUDA_NO_HALF2_OPERATORS__',
    '--expt-relaxed-constexpr',
    '--extended-lambda',
]

# Define the arguments exclusive to GCC
gcc_cuda_args = [
    '--compiler-options=-fPIC',
    '--compiler-options=-pipe',
    '--compiler-options=-Wno-unknown-pragmas',
]
gcc_cpp_args = []

# Define the arguments exclusive to msvc
msvc_cuda_args = [
    '--compiler-options=/wd4819',
    '--compiler-options=/wd4251',
    '--compiler-options=/wd4244',
    '--compiler-options=/wd4267',
    '--compiler-options=/wd4275',
    '--compiler-options=/wd4018',
    '--compiler-options=/wd4190',
    '--compiler-options=/wd4624',
    '--compiler-options=/wd4067',
    '--compiler-options=/wd4068',
    '--compiler-options=/Zc:preprocessor',
    '-Xcudafe',
    '--diag_suppress=base_class_has_different_dll_interface',
    '-Xcudafe',
    '--diag_suppress=field_without_dll_interface',
    '-Xcudafe',
    '--diag_suppress=dll_interface_conflict_none_assumed',
    '-Xcudafe',
    '--diag_suppress=dll_interface_conflict_dllexport_assumed',
]
msvc_cpp_args = [
    '/wd4819',
    '/wd4251',
    '/wd4244',
    '/wd4267',
    '/wd4275',
    '/wd4018',
    '/wd4190',
    '/wd4624',
    '/wd4067',
    '/wd4068',
]

# Get and validate our compilers
cpp_compiler = meson.get_compiler('cpp')
if cpp_compiler.get_id() not in cpp_compilers
    error('C++ compiler "' + cpp_compiler.get_id() + '" is not supported')
endif
cuda_compiler = meson.get_compiler('cuda')
if cuda_compiler.get_id() not in cuda_compilers
    error('CUDA C++ compiler "' + cuda_compiler.get_id() + '" is not supported')
endif

# Get the CUDA dependency
cuda_dep = dependency('cuda', version: cuda_target, modules: cuda_modules)

# Create the PyTorch dependency. For this we need to find the platform-specific Python installation directory where
# PyTorch is installed. Then we can find the expected libraries and headers and define our custom dependency.
py_platlib = py.get_path('platlib')
torch_includes = []
foreach torch_include: base_torch_include_paths
    torch_includes += py_platlib / torch_include
endforeach
torch_library_path = py_platlib / base_torch_library_path
torch_libraries = []
foreach torch_library_name: torch_library_names
    torch_libraries += cpp_compiler.find_library(torch_library_name, dirs: torch_library_path, required: true)
endforeach
torch_dep = declare_dependency(
    include_directories: torch_includes,
    dependencies: [torch_libraries, cuda_dep]
)

## Now it's time to build up our compiler options

# The first step is to make sure out CUDA compiler targets the same C++ standard as the C++ compiler
cuda_args += '-std=' + get_option('cpp_std')

# Then we'll get the gencode arguments
cuda = import('unstable-cuda')
cuda_args += cuda.nvcc_arch_flags(cuda_dep.version(), cuda_arch_list)

# Add the common defines to both the C++ and CUDA C++ argument lists
cpp_args += common_defines
cuda_args += common_defines

# Add the platform-specific arguments
if cpp_compiler.get_id() == 'gcc'
    cpp_args += gcc_cpp_args
    cuda_args += gcc_cuda_args
elif cpp_compiler.get_id() == 'msvc'
    cpp_args += msvc_cpp_args
    cuda_args += msvc_cuda_args
endif

# Build the extension module
py.extension_module(
    native_so_module,
    native_source_files,
    dependencies: [cuda_dep, torch_dep],
    cuda_args: cuda_args,
    cpp_args: cpp_args,
    link_language: 'cpp',
    limited_api: python_stable_abi_target,
    subdir: package_directory,
    install: true,
)

# Install the standard python source files
py.install_sources(python_source_files, subdir: package_directory)
