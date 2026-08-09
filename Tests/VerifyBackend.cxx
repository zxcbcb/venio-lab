// Deterministic CPU-vs-GPU backend check for Venio.
// Builds a small fixed network (no random init used — weights are overwritten
// with deterministic values), runs one forward + one backward pass and prints
// checksums. Compile once with the CPU backend and once with VENIO_USE_GPU; the
// printed numbers must match, which proves the CUDA/cuBLAS path is correct.
#include "Venio.hxx"
#include <iostream>
#include <iomanip>

int main()
{
    Eigen::setNbThreads(1);

    LinearFunction   linear;
    LogisticFunction logistic;
    SquareErrorFunction square;

    std::vector<std::shared_ptr<Layer>> layers{
        std::make_shared<SequentialLayer>(3, &linear),   // input  (3)
        std::make_shared<SequentialLayer>(4, &logistic), // hidden (4)
        std::make_shared<SequentialLayer>(2, &linear),   // output (2)
    };
    Model net(&square, layers);

    // Deterministic weights/biases (overwrite the random init).
    Matrixd w1(3, 4);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 4; ++j)
            w1(i, j) = 0.10 * (i + 1) + 0.01 * (j + 1);
    Matrixd b1(1, 4);
    for (int j = 0; j < 4; ++j) b1(0, j) = 0.05 * (j + 1);

    Matrixd w2(4, 2);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j)
            w2(i, j) = 0.20 * (i + 1) - 0.03 * (j + 1);
    Matrixd b2(1, 2);
    for (int j = 0; j < 2; ++j) b2(0, j) = 0.02 * (j + 1);

    net.setLayerWeights(1, w1);
    net.setLayerBias(1, b1);
    net.setLayerWeights(2, w2);
    net.setLayerBias(2, b2);

    Matrixd input(1, 3);
    input << 0.5, -0.2, 0.8;
    net.setInput(input);

    net.forwardPropogation();
    Matrixd out = net.getOutput();

    Matrixd target(1, 2);
    target << 0.3, 0.7;
    net.backPropogation(target);

    Matrixd g1 = net.getLayerWeightsGradient(1);
    Matrixd g2 = net.getLayerWeightsGradient(2);

#ifdef GPU_OPTIMIZATION
    std::cout << "BACKEND GPU\n";
#else
    std::cout << "BACKEND CPU\n";
#endif
    std::cout << std::fixed << std::setprecision(8);
    std::cout << "OUTPUT_SUM " << out.sum() << "\n";
    std::cout << "OUTPUT     " << out(0, 0) << " " << out(0, 1) << "\n";
    std::cout << "GRAD1_SUM  " << g1.sum() << "\n";
    std::cout << "GRAD2_SUM  " << g2.sum() << "\n";
    std::cout << "LOSS       " << net.getAverageLoss(target) << "\n";
    return 0;
}
