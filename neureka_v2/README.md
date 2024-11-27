# Neureka

## Docs

Gvsoc model repo [link](https://github.com/gvsoc/gvsoc-pulp/tree/prasadar/siracusa/pulp/neureka).

## Implemented features

- [x] Convolution w/ kernel shape 1x1
- [x] Convolution w/ kernel shape 3x3
- [x] Depthwise convolution w/ kernel shape 3x3
- [ ] Normalization and quantization
    - [x] With
    - [x] Without
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
    - [x] int32
- [x] Scale type
    - [x] uint8
    - [x] uint32
- [x] Bias type
    - [x] int32
- [ ] Weight type
    - [x] int8
    - [ ] int2-7
- [x] Dedicated weight memory
