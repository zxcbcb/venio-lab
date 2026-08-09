// Correctness test for ConvolutionLayer: verifies the analytic kernel gradient
// against a finite-difference numerical gradient. Loss = sum(activated output),
// so d(Loss)/d(activated) = 1 everywhere (grad_output = ones).
#include "Venio.hxx"
#include <iostream>
#include <iomanip>
#include <cmath>

static double lossOf(ConvolutionLayer &c, const Matrixd &input)
{
    c.propogateLayer(input);
    return c.getLayerActiveValues().sum();
}

int main()
{
    Eigen::setNbThreads(1);
    LogisticFunction logistic;

    ConvolutionLayer conv(2, 2, &logistic);
    Matrixd K(2, 2);
    K << 0.20, -0.10,
         0.30,  0.05;
    conv.setKernel(K);
    conv.setConvBias(0.10);

    Matrixd input(3, 3);
    input << 0.1, 0.2, 0.3,
             0.4, 0.5, 0.6,
             0.7, 0.8, 0.9;

    conv.propogateLayer(input);
    Matrixd out = conv.getLayerActiveValues(); // 2x2

    Matrixd gout = Matrixd::Ones(out.rows(), out.cols());
    Matrixd dinput = conv.backwardLayer(gout, input);
    Matrixd analytic = conv.getKernelGradient();

    const double eps = 1e-6;
    Matrixd numeric(2, 2);
    for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
        {
            Matrixd Kp = K; Kp(a, b) += eps; conv.setKernel(Kp); conv.setConvBias(0.10);
            double lp = lossOf(conv, input);
            Matrixd Km = K; Km(a, b) -= eps; conv.setKernel(Km); conv.setConvBias(0.10);
            double lm = lossOf(conv, input);
            numeric(a, b) = (lp - lm) / (2 * eps);
        }
    conv.setKernel(K);

    double maxdiff = (analytic - numeric).cwiseAbs().maxCoeff();

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "CONV_OUT_SUM  " << out.sum() << "\n";
    std::cout << "DINPUT_SUM    " << dinput.sum() << "\n";
    std::cout << "ANALYTIC_GRAD " << analytic(0,0) << " " << analytic(0,1) << " " << analytic(1,0) << " " << analytic(1,1) << "\n";
    std::cout << "NUMERIC_GRAD  " << numeric(0,0) << " " << numeric(0,1) << " " << numeric(1,0) << " " << numeric(1,1) << "\n";
    std::cout << "MAX_ABS_DIFF  " << maxdiff << "\n";
    std::cout << (maxdiff < 1e-5 ? "GRADCHECK PASS\n" : "GRADCHECK FAIL\n");
    return 0;
}
