# PULP-NNX kernel libraries

A kernel library targeting Neural Network Accelerators developed in the PULP group.

## **Disclaimer**

This library is considered unstable and might go through major changes until a stable release of v1.0.0.

## Repository structure

- inc: interface to the library
- src: implementations of the interface for specific accelerators
- util: utilities used by all the accelerators
- &lt;accelerator>:
    - hal: hardware abstraction layer
    - gvsoc: gvsoc-specific functions
- test: testing folder ([more info](test/README.md))

## Accelerators

### NE16

Github repo [link](https://github.com/pulp-platform/ne16).

#### Implemented features

- [x] Convolution w/ kernel shape 1x1
- [x] Convolution w/ kernel shape 3x3
- [x] Depthwise convolution w/ kernel shape 3x3
- [x] Stride 1x1
- [x] Stride 2x2
- [ ] Normalization and quantization
    - [x] With
    - [ ] Without
    - [x] Relu (w/ and w/o)
    - [x] Bias (w/ and w/o)
    - [ ] Per-channel shift
    - [x] Per-layer shift
    - [ ] Rounding
- [ ] Input type
    - [x] uint8
    - [ ] uint16
- [ ] Output type
    - [x] int8
    - [x] uint8 (only w/ Relu)
    - [ ] int32
    - [ ] uint32 (only w/ Relu)
- [ ] Scale type
    - [x] uint8
    - [ ] uint16
    - [ ] uint32
- [x] Bias type
    - [x] int32
- [ ] Weight type
    - [x] int8
    - [ ] int2-7

### Neureka

**Untested and considered broken.**

## Testing

You can find information about testing in the dedicated [README](test/README.md).

## Contributing

Bug reports and feature requests should be reported through issues.
All the development should be done through forks and merged onto the `dev` branch with pull requests.

## Versioning

The library will follow the [Semantic Versioning](https://semver.org/).

## Citing

*TBA*

## Contributors

* Luka Macan <[luka.macan@unibo.it](mailto:luka.macan@unibo.it)>
* Francesco Conti <[fconti@unibo.it](mailto:fconti@unibo.it)>
* Arpan Prasad <[prasadar@iis.ee.ethz.ch](mailto:prasadar@iis.ee.ethz.ch)>

## License

Licensed under Apache-2.0; the whole text of the license can be found in the [LICENSE](LICENSE) file.
