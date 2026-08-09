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
- [x] **Transformer** block — single-head self-attention (forward+backward, gradient-checked) + encoder block (attention→residual→LayerNorm→FFN→residual→LayerNorm) forward verified
- [x] **Test layers** on noise and on images — Conv edge-filter verified on random noise and on a real photo (lena.jpg → saved edge map)
- [x] **Train an end-to-end model** on image data — coordinate-MLP fits a photo via GD + SquareError, loss 0.19 → 0.0024

## Advanced layers (extended, gradient-checked)

Beyond the roadmap, the library now includes production-shaped layers, each with a
full backward pass verified numerically (analytic vs finite-difference, diff < 1e-9):

- **`Conv2D`** — multi-channel input, multiple filters, stride, padding, full backward
- **`MultiHeadAttention`** — multi-head self-attention with output projection, full backward
- **`TransformerEncoder`** — complete block backward: LayerNorm + FFN + residuals + MHA
- **`MaxPool2D`** — max pooling with gradient routing; **`Flatten`** — conv↔dense bridge

### Test suite (`Tests/`, each an `add_executable` in CMakeLists)

| Test | Checks |
|---|---|
| `VerifyBackend` | dense forward/backward, CPU output == GPU output (bit-identical) |
| `ConvTest` | single-channel conv gradient check |
| `Conv2DTest` | general Conv2D gradient check (channels/filters/stride/pad) |
| `AttnTest` / `MHATest` | single-head / multi-head attention gradient checks |
| `TransformerTest` / `EncoderTest` | transformer block forward / full-block backward |
| `PoolTest` | MaxPool2D gradient check + Flatten round-trip |
| `LayerImageTest` | conv on noise and on a real photo |
| `TrainImage` | end-to-end training on a photo |
| `OptCompare` | GD vs ADAM vs RMSProp vs Adagrad vs Adadelta |
| `Bench` | CPU vs GPU timing |

## Authors

pixaut · drgnbon

## License

MIT — see [LICENSE](LICENSE).
