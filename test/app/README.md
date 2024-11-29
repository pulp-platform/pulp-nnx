# Test application

## Build

### CMake

For the cmake build you have to specify the toolchain file with the `-DCMAKE_TOOLCHAIN_FILE` flag:
```
cmake -S . -B build -G Ninja -DACCELERATOR=<accelerator> -DCMAKE_TOOLCHAIN_FILE=cmake/toolchain_<toolchain>.cmake
```

After that you can build the application by running:
```
cmake --build build
```

No need to regenerate the project (1st cmake command) after altering the sources, just rerun the build.

## TODO

- environment variables that need to be set
- variables that need to be passed to the build (ACCELERATOR)
- the make build flow
