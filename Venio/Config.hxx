#pragma once

#include <Eigen\Core> // need for all classes maybe
#include "Kernel.hxx"

typedef Eigen::MatrixXd Matrixd;
typedef Eigen::ArrayXXd Arrayd;
typedef Eigen::VectorXd Vectord;
namespace K = Kernel;

// ---- Backend selection --------------------------------------------------
// Define VENIO_USE_GPU (e.g. CMake: -DVENIO_USE_GPU) to run matrix ops on the
// CUDA/cuBLAS kernel; otherwise the CPU (Eigen) path is used.
// This switch lives in Config.hxx — a header every translation unit includes
// (Model, SequentialLayer, optimizers) — so all of them agree on the backend.
// Putting it in a single .cxx (as it was in Main.cxx) left the library TUs
// blind to it, which is why the GPU path never actually ran.
#if defined(VENIO_USE_GPU)
    #ifndef GPU_OPTIMIZATION
        #define GPU_OPTIMIZATION
    #endif
#else
    #ifndef CPU_OPTIMIZATION
        #define CPU_OPTIMIZATION
    #endif
#endif