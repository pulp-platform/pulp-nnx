# PULP-NNX kernel libraries

A kernel library targeting Neural Network Accelerators developed in the PULP group.

## **Disclaimer**

This library is considered unstable and might go through major changes until a stable release of v1.0.0.

## NNX Interface

The interface to each accelerator consists of these functions:

```c
void <accelerator>_nnx_init(<accelerator>_dev_t *dev, <accelerator>_pulp_conf_t *conf);
void <accelerator>_nnx_term(<accelerator>_dev_t *dev);
int <accelerator>_nnx_dispatch(<accelerator>_dev_t *dev, <accelerator>_task_t *task);
int <accelerator>_nnx_dispatch_check(<accelerator>_dev_t *dev);
void <accelerator>_nnx_dispatch_wait(<accelerator>_dev_t *dev);
int <accelerator>_nnx_resolve_check(<accelerator>_dev_t *dev, <accelerator>_task_t *task);
void <accelerator>_nnx_resolve_wait(<accelerator>_dev_t *dev, <accelerator>_task_t *task);
```

Each accelerator has their own named function in case there exist multiple types of accelerators on a same board.

Each function accepts a pointer to a `<accelerator>_dev_t` type which to discern between each accelerator.

_Note: The accelerator can provide additional helper functions if needed._

## Repository structure

- inc: nnx interface for each accelerator
- src: implementation for each accelerator
- util: utilities used by all the accelerators
- &lt;accelerator>:
    - hal: hardware abstraction layer
    - gvsoc: gvsoc-specific functions
    - bsp: board support package for each board that has the accelerator
- test: testing folder ([more info](test/README.md))

## Accelerators

- [NE16](ne16/README.md)
- [Neureka](neureka/README.md)

## Testing

You can find information about testing in the dedicated [README](test/README.md).

### Environment

The library was tested with following pairs of SDKs and compilers:

| SDK | SDK Commit Hash | Compiler | Compiler Commit Hash |
| --- | --------------- | -------- | -------------------- |
| gap\_sdk (obtainable from GreenWaves Technologies) | 90df4ce219 | [gap\_gnu\_toolchain](https://github.com/GreenWaves-Technologies/gap_gnu_toolchain) | 360fd4f9d6 |
| [pulp-sdk](https://github.com/Scheremo/pulp-sdk) | c216298881 | [pulp-riscv-gnu-toolchain](https://github.com/pulp-platform/pulp-riscv-gnu-toolchain) | 9938bd8fcf (release v1.0.16) |

## Contributing

Bug reports and feature requests should be reported through issues.
All the development should be done through forks and merged onto the `dev` branch with pull requests.

## Versioning

The library will follow the [Semantic Versioning](https://semver.org/).

## Publication

<details>
<summary>If you use PULP-NNX in your work, you can cite us:</summary>

```
@inproceedings{10.1145/3607889.3609092,
    author = {Macan, Luka and Burrello, Alessio and Benini, Luca and Conti, Francesco},
    title = {WIP: Automatic DNN Deployment on Heterogeneous Platforms: the GAP9 Case Study},
    year = {2024},
    isbn = {9798400702907},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3607889.3609092},
    doi = {10.1145/3607889.3609092},
    abstract = {Emerging Artificial-Intelligence-enabled System-on-Chips (AI-SoCs) combine a flexible microcontroller with parallel Digital Signal Processors (DSP) and heterogeneous acceleration capabilities. In this Work-in-Progress paper, we focus on the GAP9 RISC-V SoC as a case study to show how the open-source DORY Deep Neural Network (DNN) tool flow can be extended for heterogeneous acceleration by fine grained interleaving of a dedicated Neural Engine and a cluster of RISC-V cores. Our results show that up to 91\% of the peak accelerator throughput can be extracted in end-to-end execution of benchmarks based on MobileNet-V1 and V2.},
    booktitle = {Proceedings of the International Conference on Compilers, Architecture, and Synthesis for Embedded Systems},
    pages = {9–10},
    numpages = {2},
    keywords = {TinyML, MCUs, deep learning, HW accelerators},
    location = {<conf-loc>, <city>Hamburg</city>, <country>Germany</country>, </conf-loc>},
    series = {CASES '23 Companion}
}
```

</details>

## Contributors

* Luka Macan <[luka.macan@unibo.it](mailto:luka.macan@unibo.it)>
* Francesco Conti <[fconti@unibo.it](mailto:fconti@unibo.it)>
* Arpan Suravi Prasad <[prasadar@iis.ee.ethz.ch](mailto:prasadar@iis.ee.ethz.ch)>

## License

Licensed under Apache-2.0; the whole text of the license can be found in the [LICENSE](LICENSE) file.
