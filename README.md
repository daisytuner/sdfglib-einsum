# sdfglib-einsum

## Requirements

Build and install sdfglib. Set the `-DCMAKE_INSTALL_PREFIX=` accordingly.

```bash
git clone git@github.com:daisytuner/sdfglib.git
cd sdfglib

mkdir build && cd build
cmake -G Ninja -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Debug  ..
ninja -j$(nproc)
ninja install
```

## Build

In case sdfglib was not installed to a standard search path of the build tools (e.g., /usr/local), set the search paths for the cmake configs, such that cmake finds sdfglib and symengine:

```bash
mkdir build && cd build
cmake -G Ninja -DCMAKE_C_COMPILER=clang-19 -DCMAKE_CXX_COMPILER=clang++-19 -DCMAKE_BUILD_TYPE=Debug -Dsdfglib_DIR=<path-to-sdfglib-install-lib-cmake-sdfglib> -DSymEngine_DIR=<path-to-sdfglib-install-lib-cmake-symengine> ..
ninja -j$(nproc)
```
