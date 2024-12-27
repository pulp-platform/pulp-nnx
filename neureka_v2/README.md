# Neureka v2

## Docs

Gvsoc model repo [link](https://github.com/lukamac/gvsoc-pulp/tree/fix-vectorload).

## Implemented features

- [x] Convolution w/ kernel shape 1x1
- [x] Convolution w/ kernel shape 3x3
- [x] Depthwise convolution w/ kernel shape 3x3
- [ ] Normalization and quantization
    - [x] With
    - [ ] Without
    - [x] Relu (w/ and w/o)
    - [x] Bias (w/ and w/o)
    - [ ] Per-channel shift
    - [x] Per-layer shift
- [x] Input type
    - [x] uint8
    - [x] int8
- [x] Output type
    - [x] int8
    - [x] uint8 (only w/ Relu)
    - [ ] int32
- [x] Scale type
    - [x] int8
    - [ ] int32
- [x] Bias type
    - [x] int32
- [ ] Weight type
    - [x] int8
    - [ ] int2-7
- [ ] Weight memory
    - [ ] Shared TCDM
    - [x] Private SRAM
    - [x] Private MRAM
