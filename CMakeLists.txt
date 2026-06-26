cmake_minimum_required(VERSION 3.26)
project(sabc_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

# mlx is a namespace package (mlx.__file__ is None), so locate it via
# the site-packages directory of the active interpreter.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE SITE_PKG_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${SITE_PKG_DIR}/mlx")
find_package(MLX CONFIG REQUIRED)

# Drop STABLE_ABI so nanobind cross-module type sharing (needed for
# mx::array interop in Task 4) is not blocked by the limited ABI.
nanobind_add_module(_core NB_STATIC
  csrc/bindings.cpp
  csrc/cdf.cpp
  csrc/distance.cpp
  csrc/epsilon.cpp
  csrc/resample.cpp
)
target_include_directories(_core PRIVATE csrc)
target_link_libraries(_core PRIVATE mlx)
target_compile_features(_core PRIVATE cxx_std_23)

# Embed an rpath so the extension finds libmlx.dylib at runtime without
# requiring LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
set_target_properties(_core PROPERTIES
  INSTALL_RPATH "${SITE_PKG_DIR}/mlx/lib"
  BUILD_RPATH   "${SITE_PKG_DIR}/mlx/lib"
)

install(TARGETS _core LIBRARY DESTINATION sabc)
