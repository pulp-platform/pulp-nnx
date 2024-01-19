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
* Arpan Suravi Prasad <[prasadar@iis.ee.ethz.ch](mailto:prasadar@iis.ee.ethz.ch)>

## License

Licensed under Apache-2.0; the whole text of the license can be found in the [LICENSE](LICENSE) file.
