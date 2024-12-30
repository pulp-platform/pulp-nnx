# Test application

## Build

There are two build flows given for the test app, the Make one and the CMake one.
Both flows use a flag `ACCELERATOR` to decide which accelerator to build the application for.
Choices are _neureka_ and _neureka_v2_.

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

## TODO

- environment variables that need to be set
