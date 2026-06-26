cmake_minimum_required(VERSION 3.26)
project(sabc_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SABC_BUILD_TESTS "Build native C++ unit tests" OFF)

# Python locates MLX (and nanobind for the extension). The extension also needs
# the Development.Module component; the C++ tests need only the interpreter.
if(SABC_BUILD_TESTS)
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter)
else()
  find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
endif()

# mlx is a namespace package (mlx.__file__ is None), so locate it via
# the site-packages directory of the active interpreter.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
    "import sysconfig; print(sysconfig.get_paths()['purelib'])"
  OUTPUT_VARIABLE SITE_PKG_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${SITE_PKG_DIR}/mlx")
find_package(MLX CONFIG REQUIRED)

# Algorithm sources shared by the extension and the C++ tests (everything
# except bindings.cpp, which needs nanobind).
set(SABC_CORE_SOURCES
  src/cpp/cdf.cpp
  src/cpp/distance.cpp
  src/cpp/epsilon.cpp
  src/cpp/proposals.cpp
  src/cpp/resample.cpp
  src/cpp/sabc.cpp
)

if(SABC_BUILD_TESTS)
  # Fetch doctest's source but skip its CMake project (SOURCE_SUBDIR none) --
  # it is header-only and its CMakeLists requires an ancient CMake.
  include(FetchContent)
  FetchContent_Declare(doctest
    GIT_REPOSITORY https://github.com/doctest/doctest.git
    GIT_TAG v2.4.11
    SOURCE_SUBDIR none)
  FetchContent_MakeAvailable(doctest)

  file(GLOB SABC_TEST_SOURCES tests/cpp/*.cpp)
  add_executable(sabc_tests ${SABC_TEST_SOURCES} ${SABC_CORE_SOURCES})
  target_include_directories(sabc_tests PRIVATE src/cpp ${doctest_SOURCE_DIR})
  target_link_libraries(sabc_tests PRIVATE mlx)
  target_compile_features(sabc_tests PRIVATE cxx_std_23)
  set_target_properties(sabc_tests PROPERTIES
    BUILD_RPATH "${SITE_PKG_DIR}/mlx/lib")
  enable_testing()
  add_test(NAME sabc_tests COMMAND sabc_tests)
else()
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE)
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
  find_package(nanobind CONFIG REQUIRED)

  # Drop STABLE_ABI so nanobind cross-module type sharing (needed for
  # mx::array interop) is not blocked by the limited ABI.
  nanobind_add_module(_core NB_STATIC
    src/cpp/bindings.cpp
    ${SABC_CORE_SOURCES}
  )
  target_include_directories(_core PRIVATE src/cpp)
  target_link_libraries(_core PRIVATE mlx)
  target_compile_features(_core PRIVATE cxx_std_23)

  # Embed an rpath so the extension finds libmlx.dylib at runtime without
  # requiring LD_LIBRARY_PATH / DYLD_LIBRARY_PATH.
  set_target_properties(_core PROPERTIES
    INSTALL_RPATH "${SITE_PKG_DIR}/mlx/lib"
    BUILD_RPATH   "${SITE_PKG_DIR}/mlx/lib"
  )

  install(TARGETS _core LIBRARY DESTINATION sabc)
endif()
