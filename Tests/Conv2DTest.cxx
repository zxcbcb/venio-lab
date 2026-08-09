// Gradient check for the general Conv2D (multi-channel, multi-filter, stride, pad).
// Loss = sum of all output channels; analytic kernel grads vs central differences.
#include "Conv2D.hxx"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>

static double lossOf(Conv2D &c, const Tensor &in)
{
    Tensor o = c.forward(in);
    double s = 0.0;
    for (auto &m : o) s += m.sum();
    return s;
}

static double checkConfig(int cin, int cout, int kh, int kw, int stride, int pad,
                          int H, int W, const char *name)
{
    Conv2D conv(cin, cout, kh, kw, stride, pad);

    for (int o = 0; o < cout; ++o)
        for (int i = 0; i < cin; ++i)
        {
            Matrixd K(kh, kw);
            for (int a = 0; a < kh; ++a)
                for (int b = 0; b < kw; ++b)
                    K(a, b) = 0.1 * std::sin(a + 2.0 * b + 3.0 * i + 5.0 * o);
            conv.setW(o, i, K);
            conv.setBias(o, 0.05 * (o + 1));
        }

    Tensor in(cin);
    for (int i = 0; i < cin; ++i)
    {
        Matrixd X(H, W);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                X(y, x) = 0.1 * std::sin(y + 1.3 * x + 0.7 * i);
        in[i] = X;
    }

    Tensor out = conv.forward(in);
    Tensor dout(cout);
    for (int o = 0; o < cout; ++o)
        dout[o] = Matrixd::Ones(out[o].rows(), out[o].cols());
    conv.backward(dout);

    const double eps = 1e-6;
    double maxdiff = 0.0;
    for (int o = 0; o < cout; ++o)
        for (int i = 0; i < cin; ++i)
        {
            Matrixd analytic = conv.getGradW(o, i);
            Matrixd K = conv.getW(o, i);
            for (int a = 0; a < kh; ++a)
                for (int b = 0; b < kw; ++b)
                {
                    Matrixd Kp = K; Kp(a, b) += eps; conv.setW(o, i, Kp);
                    double lp = lossOf(conv, in);
                    Matrixd Km = K; Km(a, b) -= eps; conv.setW(o, i, Km);
                    double lm = lossOf(conv, in);
                    conv.setW(o, i, K);
                    double num = (lp - lm) / (2 * eps);
                    maxdiff = std::max(maxdiff, std::abs(num - analytic(a, b)));
                }
        }

    const int Ho = (H + 2 * pad - kh) / stride + 1;
    const int Wo = (W + 2 * pad - kw) / stride + 1;
    std::cout << std::left << std::setw(24) << name << std::right
              << " out=" << out[0].rows() << "x" << out[0].cols()
              << " (exp " << Ho << "x" << Wo << ") chans=" << (int)out.size()
              << " max_grad_diff=" << std::scientific << std::setprecision(2) << maxdiff
              << (maxdiff < 1e-5 ? "  PASS" : "  FAIL") << "\n";
    return maxdiff;
}

int main()
{
    Eigen::setNbThreads(1);
    double m1 = checkConfig(2, 3, 3, 3, 1, 1, 5, 5, "cin2 cout3 k3 s1 p1");
    double m2 = checkConfig(3, 2, 3, 3, 2, 0, 7, 7, "cin3 cout2 k3 s2 p0");
    double m3 = checkConfig(1, 4, 2, 2, 2, 1, 6, 6, "cin1 cout4 k2 s2 p1");
    double worst = std::max(m1, std::max(m2, m3));
    std::cout << (worst < 1e-5 ? "CONV2D GRADCHECK PASS\n" : "CONV2D GRADCHECK FAIL\n");
    return 0;
}
