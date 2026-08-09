// Tests MaxPool2D (numeric gradient check via max-routing) and Flatten (round-trip).
#include "Pool.hxx"
#include <iostream>
#include <iomanip>
#include <cmath>

static double poolLoss(MaxPool2D &p, const Tensor &in)
{
    Tensor o = p.forward(in);
    double s = 0.0;
    for (auto &m : o) s += m.sum();
    return s;
}

int main()
{
    Eigen::setNbThreads(1);
    const int C = 2, H = 4, W = 4, k = 2, stride = 2;

    Tensor in(C);
    for (int c = 0; c < C; ++c)
    {
        Matrixd X(H, W);
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
                X(y, x) = std::sin(1.7 * y + 2.3 * x + 5.0 * c); // distinct -> no ties
        in[c] = X;
    }

    MaxPool2D pool(k, stride);
    Tensor out = pool.forward(in);
    Tensor dOut(C);
    for (int c = 0; c < C; ++c) dOut[c] = Matrixd::Ones(out[c].rows(), out[c].cols());
    Tensor din = pool.backward(dOut);

    // numeric gradient check on the input
    const double eps = 1e-6;
    double maxdiff = 0.0;
    for (int c = 0; c < C; ++c)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x)
            {
                Tensor ip = in; ip[c](y, x) += eps; double lp = poolLoss(pool, ip);
                Tensor im = in; im[c](y, x) -= eps; double lm = poolLoss(pool, im);
                double num = (lp - lm) / (2 * eps);
                maxdiff = std::max(maxdiff, std::abs(num - din[c](y, x)));
            }
    // restore pool caches to the original forward
    pool.forward(in);

    std::cout << "MAXPOOL out=" << out[0].rows() << "x" << out[0].cols() << " chans=" << (int)out.size()
              << " max_grad_diff=" << std::scientific << std::setprecision(2) << maxdiff
              << (maxdiff < 1e-5 ? "  PASS" : "  FAIL") << "\n";

    // Flatten round-trip
    Flatten fl;
    Matrixd flat = fl.forward(in);
    Tensor back = fl.backward(flat);
    double rterr = 0.0;
    for (int c = 0; c < C; ++c) rterr = std::max(rterr, (back[c] - in[c]).cwiseAbs().maxCoeff());
    bool sizeOK = (flat.rows() == 1 && flat.cols() == C * H * W);
    std::cout << "FLATTEN shape=1x" << flat.cols() << " (exp 1x" << C * H * W << ")"
              << " roundtrip_err=" << std::scientific << rterr
              << ((sizeOK && rterr < 1e-12) ? "  PASS" : "  FAIL") << "\n";

    bool ok = (maxdiff < 1e-5) && sizeOK && (rterr < 1e-12);
    std::cout << (ok ? "POOL/FLATTEN TEST PASS\n" : "POOL/FLATTEN TEST FAIL\n");
    return ok ? 0 : 1;
}
