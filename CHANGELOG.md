# Changelog

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
