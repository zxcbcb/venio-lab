// Smoke test for TransformerBlock forward: deterministic weights, checks output
// shape and that the final LayerNorm gives rows with mean~0 and variance~1.
#include "TransformerBlock.hxx"
#include <iostream>
#include <iomanip>
#include <cmath>

int main()
{
    Eigen::setNbThreads(1);
    const int d = 4, hidden = 8, seq = 5;

    TransformerBlock block(d, hidden);

    Matrixd Wq(d, d), Wk(d, d), Wv(d, d), W1(d, hidden), W2(hidden, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
        {
            Wq(i, j) = 0.05 * (i + 1) + 0.01 * (j + 1);
            Wk(i, j) = 0.03 * (i + 1) - 0.02 * (j + 1);
            Wv(i, j) = -0.02 * (i + 1) + 0.04 * (j + 1);
        }
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < hidden; ++j)
            W1(i, j) = 0.02 * (i + 1) + 0.005 * (j + 1);
    for (int i = 0; i < hidden; ++i)
        for (int j = 0; j < d; ++j)
            W2(i, j) = -0.01 * (i + 1) + 0.03 * (j + 1);

    block.attention().setWq(Wq);
    block.attention().setWk(Wk);
    block.attention().setWv(Wv);
    block.setW1(W1);
    block.setW2(W2);
    block.setB1(Eigen::RowVectorXd::Zero(hidden));
    block.setB2(Eigen::RowVectorXd::Zero(d));

    Matrixd X(seq, d);
    for (int i = 0; i < seq; ++i)
        for (int j = 0; j < d; ++j)
            X(i, j) = 0.1 * (i + 1) - 0.05 * (j + 1);

    Matrixd out = block.forward(X);

    double maxmean = 0.0, maxvarerr = 0.0;
    for (int i = 0; i < out.rows(); ++i)
    {
        double m = out.row(i).mean();
        double var = (out.row(i).array() - m).square().sum() / out.cols();
        maxmean = std::max(maxmean, std::abs(m));
        maxvarerr = std::max(maxvarerr, std::abs(var - 1.0));
    }

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "OUT_SHAPE " << out.rows() << "x" << out.cols() << "\n";
    std::cout << "OUT_SUM   " << out.sum() << "\n";
    std::cout << "MAX_ROW_MEAN   " << maxmean << "\n";
    std::cout << "MAX_VAR_ERR    " << maxvarerr << "\n";
    bool ok = (out.rows() == seq && out.cols() == d && maxmean < 1e-6 && maxvarerr < 1e-2);
    std::cout << (ok ? "TRANSFORMER FORWARD OK\n" : "TRANSFORMER FORWARD FAIL\n");
    return 0;
}
