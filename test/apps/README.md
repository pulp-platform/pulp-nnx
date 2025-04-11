# Test applications

There are 2 testing applications:
1. pulp-nnx - tests the whole library
2. pulp-nnx-hal - tests just the hardware abstraction layer parts

Both apps are very similar except for the fact that the Make build flow is not implemented for the pulp-nnx-hal app.

## Build

There are two build flows given for the test app, the Make one and the CMake one.
Both flows use a flag `ACCELERATOR` to decide which accelerator to build the application for.
Choices are _neureka_ and _neureka_v2_.

### Environment variables

Both build fows expect some environment variables to be set and will fail if they are not set:
- `GVSOC` - path to the `gvsoc` executable (not the directory)
- `TOOLCHAIN_LLVM_INSTALL_DIR` - path to the llvm install directory
- `PULP_SDK_HOME` - path to the pulp-sdk directory

### Make

For the Make flow you need to specify the `ACCELERATOR` flag either as an environment variable
```
export ACCELERATOR=<accelerator>
```
or you can add it to the make command
```
ACCELERATOR=<accelerator> make clean all run
```

### CMake

For the cmake build you have to specify the `ACCELERATOR` flag with `-DACCLERATOR=<accelerator> and the toolchain file with the `-DCMAKE_TOOLCHAIN_FILE` flag:
```
cmake -S . -B build -G Ninja -DACCELERATOR=<accelerator> -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain_<toolchain>.cmake
```

After that you can build the application by running:
```
cmake --build build
```

No need to regenerate the project (1st cmake command) after altering the sources, just rerun the build.
