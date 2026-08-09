// Gradient check for MultiHeadAttention. Loss = sum(Out), dOut = ones.
// Verifies analytic grads for Wq, Wk, Wv, Wo vs central finite differences.
#include "MultiHeadAttention.hxx"
#include <iostream>
#include <iomanip>

static double lossOf(MultiHeadAttention &a, const Matrixd &X)
{
    return a.forward(X).sum();
}

static double gradCheck(const char *name, MultiHeadAttention &mha, const Matrixd &X,
                        const Matrixd &W, void (MultiHeadAttention::*setter)(const Matrixd &),
                        const Matrixd &analytic)
{
    const int r = (int)W.rows(), c = (int)W.cols();
    const double eps = 1e-6;
    Matrixd numeric(r, c);
    for (int i = 0; i < r; ++i)
        for (int j = 0; j < c; ++j)
        {
            Matrixd Wp = W; Wp(i, j) += eps; (mha.*setter)(Wp);
            double lp = lossOf(mha, X);
            Matrixd Wm = W; Wm(i, j) -= eps; (mha.*setter)(Wm);
            double lm = lossOf(mha, X);
            numeric(i, j) = (lp - lm) / (2 * eps);
        }
    (mha.*setter)(W);
    double md = (analytic - numeric).cwiseAbs().maxCoeff();
    std::cout << name << " max_abs_diff " << std::scientific << std::setprecision(2) << md << "\n";
    return md;
}

int main()
{
    Eigen::setNbThreads(1);
    const int d = 6, heads = 2, seq = 4;

    MultiHeadAttention mha(d, heads);

    auto fill = [&](double s) {
        Matrixd m(d, d);
        for (int i = 0; i < d; ++i)
            for (int j = 0; j < d; ++j)
                m(i, j) = s * std::sin(1.0 * i + 1.7 * j + s);
        return m;
    };
    Matrixd Wq = fill(0.11), Wk = fill(0.07), Wv = fill(0.05), Wo = fill(0.09);
    mha.setWq(Wq); mha.setWk(Wk); mha.setWv(Wv); mha.setWo(Wo);

    Matrixd X(seq, d);
    for (int i = 0; i < seq; ++i)
        for (int j = 0; j < d; ++j)
            X(i, j) = 0.1 * (i + 1) - 0.05 * (j + 1);

    Matrixd out = mha.forward(X);
    Matrixd dOut = Matrixd::Ones(seq, d);
    Matrixd dX = mha.backward(dOut);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "MHA_OUT_SHAPE " << out.rows() << "x" << out.cols() << " (heads=" << heads << ")\n";
    std::cout << "OUT_SUM " << out.sum() << "  DX_SUM " << dX.sum() << "\n";

    double m1 = gradCheck("Wq", mha, X, Wq, &MultiHeadAttention::setWq, mha.getGradWq());
    double m2 = gradCheck("Wk", mha, X, Wk, &MultiHeadAttention::setWk, mha.getGradWk());
    double m3 = gradCheck("Wv", mha, X, Wv, &MultiHeadAttention::setWv, mha.getGradWv());
    double m4 = gradCheck("Wo", mha, X, Wo, &MultiHeadAttention::setWo, mha.getGradWo());

    double worst = std::max(std::max(m1, m2), std::max(m3, m4));
    std::cout << "WORST_DIFF " << std::scientific << worst << "\n";
    std::cout << (worst < 1e-5 ? "MHA GRADCHECK PASS\n" : "MHA GRADCHECK FAIL\n");
    return 0;
}
