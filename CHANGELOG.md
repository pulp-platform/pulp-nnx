# Changelog

## [Unreleased]

### Added

- github action for testing neureka
- add NnxMapping dictionary that maps accelerator name to the accelerator specific classes
- choice of data generation method (ones, incremented, or random)
- N-EUREKA accelerator support: 3x3, 1x1, and 3x3 depthwise convolution kernels
- Support for kernels without normalization and quantization for NE16
- isort check
- publication citation
- support 32bit scale
- cmake support
- const qualifier to `<acc>_dev_t` function arguments
- support for N-EUREKA's dedicated weight memory

### Changed

- python requirements are changed into requirements-pip and requirements-conda
- conftest now passes only strings to test.py to improve readability of pytest logs
- NnxMemoryLayout is now NnxWeight and also has a method for source generation
- the `wmem` field in the test configurations is now required
- `ne16_task_init` got split into smaller parts: `ne16_task_init`, `ne16_task_set_op_to_conv`, `ne16_task_set_weight_offset`, `ne16_task_set_bits`, `ne16_task_set_norm_quant`
- strides in `ne16_task_set_strides`, `ne16_task_set_dims`, and `ne16_task_set_ptrs` are now strides between consecutive elements in that dimension
- `ne16_task_queue_size` is now `NE16_TASK_QUEUE_SIZE`
- `ne16_task_set_ptrs` split into `ne16_task_set_ptrs_conv` and `ne16_task_set_ptrs_norm_quant`

### Removed

- `k_in_stride`, `w_in_stride`, `k_out_stride`, and `w_out_stride` from `ne16_nnx_dispatch_stride2x2`
- `mode` attribute from `ne16_quant_t` structure

### Fixed

- global shift should have been of type uint8 not int32
- type conversion compiler warning

## [0.3.0] - 2024-01-14

### Added

- New Hardware Processing Engine (HWPE) device in `util/hwpe.h`
- A device structure for ne16 `ne16_dev_t` in `ne16/hal/ne16.h` which extends the hwpe device
- Test app Makefile has now an `ACCELERATOR` variable to specify which accelerator is used

### Changed

- Library functions no longer start with a generic `nnx_` prefix but with `<accelerator>_nnx_` prefix
  to allow for usage of multiple kinds of accelerators in the same system
- Decoupled board specific functionality into `ne16/bsp` which also contains constant global structures
  to the implementations of the `ne16_dev_t` structure
- Moved all task related functions (`nnx_task_set_dims*`) into `ne16/hal/ne16_task.c`
- Tests adjusted for the new interface
- Test data generation moved into source files with extern declarations to check the output from the main

### Fixed

- pyright errors
- formatting errors

## [0.2.1] - 2024-01-08

### Fixed

- Stridded 2x2 mode needed to propagate `padding_bottom` when input height is smaller then 5
- Test requirements where missing the toml dependency and

## [0.2.0] - 2023-10-25

### Added

- Added timeout parameter to conftest.py
- Added stride arguments to `nnx_task_set_dims`, `nnx_task_set_dims_stride2x2`, and `nnx_dispatch_task_stride2x2`

## [0.1.0] - 2023-09-22

### Added

- Initial release of the repository.
