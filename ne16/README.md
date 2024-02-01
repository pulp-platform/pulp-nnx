# NE16

## Docs

- Github repo [link](https://github.com/pulp-platform/ne16).

## Implemented features

- [x] Convolution w/ kernel shape 1x1
- [x] Convolution w/ kernel shape 3x3
- [x] Depthwise convolution w/ kernel shape 3x3
- [x] Stride 2x2
- [ ] Normalization and quantization
    - [x] With
    - [x] Without
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
    - [x] int32
- [ ] Scale type
    - [x] uint8
    - [ ] uint16
    - [ ] uint32
- [x] Bias type
    - [x] int32
- [ ] Weight type
    - [x] int8
    - [ ] int2-7
