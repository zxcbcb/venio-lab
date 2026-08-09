# Venio

A neural network library written from scratch in modern C++ — by **pixaut** & **drgnbon**.

Venio implements the full training stack (layers, activations, losses, optimizers)
on top of [Eigen](https://eigen.tuxfamily.org) for linear algebra, with a CUDA
backend for GPU acceleration and OpenMP for CPU parallelism.

> This is the working / development repository. Experiments and new features land
> here first, then get merged upstream.

## Features

- **Layers:** Dense (`Layer`), `SequentialLayer`, `ConvolutionLayer`
- **Activations:** ReLU, LReLU, ELU, SELU, SiLU, SoftPlus, SoftSign, Logistic,
  Linear, ArcTg, TH, Sin, Sinc, ISRU, ISRLU, Benti, GH
- **Losses:** Square Error
- **Optimizers:** GD, ADAM, RMSProp, Adagrad, Adadelta, BFGS
- **Backends:** Eigen (CPU), CUDA (`CudaKernel/Kernel.cu`), OpenMP
- **Utilities:** `Model`, `RandomGenerator`, `BenchMark`, `ErrorLogger`

## Project layout

```
Venio/
├── Main.cxx                 # entry point / playground
├── CMakeLists.txt           # CUDA + CXX build (CMake >= 3.18)
├── dependencies/
│   └── eigen-3.4.0/         # bundled linear-algebra dependency
└── Venio/
    ├── Venio.hxx            # umbrella header
    ├── Config.hxx
    ├── ActivationFunctions/
    ├── Layers/              # Layer, SequentialLayer, ConvolutionLayer
    ├── LossFunctions/
    ├── Optimizers/          # GD, ADAM, RMSProp, Adagrad, Adadelta, BFGS
    ├── Model/
    ├── CudaKernel/          # CUDA kernels (Kernel.cu / Kernel.hxx)
    ├── RandomGenerator/
    ├── BenchMark/
    └── ErrorLogger/
```

## Build

Requirements:
- CMake ≥ 3.18
- A C++ compiler (MSVC / GCC / Clang)
- CUDA Toolkit (for the GPU backend — `find_package(CUDAToolkit REQUIRED)`)
- OpenMP (optional, auto-detected)

```bash
cmake -S . -B build
cmake --build build --config Release
./build/Main            # or build\Release\Main.exe on Windows
```

Eigen is bundled in `dependencies/`, so no separate install is needed.

## Roadmap

- [x] **Fix the GPU / CUDA backend** (`CudaKernel`) — CPU/GPU switch, CUDA 12.6 (sm_61), GPU output verified bit-identical to CPU
- [x] **Convolution layer** — single-channel 2D `ConvolutionLayer` (forward+backward), gradient-checked (analytic == numeric)
- [ ] **Transformer** block
- [ ] **Test layers** on noise and on images
- [ ] **Train an end-to-end model** on image data

## Authors

pixaut · drgnbon

## License

MIT — see [LICENSE](LICENSE).
