// Timing benchmark: identical dense forward+backward loop, compiled once for the
// CPU backend and once with VENIO_USE_GPU, to compare wall-clock time.
#include "Venio.hxx"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>

int main()
{
    Eigen::setNbThreads(8);
    LinearFunction linear;
    LogisticFunction logistic;
    SquareErrorFunction square;

    std::vector<std::shared_ptr<Layer>> layers{
        std::make_shared<SequentialLayer>(512, &linear),
        std::make_shared<SequentialLayer>(1024, &logistic),
        std::make_shared<SequentialLayer>(1024, &logistic),
        std::make_shared<SequentialLayer>(256, &linear),
    };
    Model net(&square, layers);

    Matrixd a(1, 512); a.setConstant(0.1);
    net.setInput(a);
    Matrixd b(1, 256); b.setConstant(0.1);

    const int iters = 50;

    // warmup (also initializes CUDA context in GPU build)
    net.forwardPropogation();
    net.backPropogation(b);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i)
    {
        net.forwardPropogation();
        net.backPropogation(b);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

#ifdef GPU_OPTIMIZATION
    std::cout << "BACKEND GPU  ";
#else
    std::cout << "BACKEND CPU  ";
#endif
    std::cout << std::fixed << std::setprecision(3)
              << iters << " iters | total " << ms << " ms | per-iter " << ms / iters << " ms\n";
    return 0;
}
