// Gradient check for single-head self-attention. Loss = sum(Out), so dOut = ones.
// Verifies analytic gradients for Wq, Wk, Wv against central finite differences.
#include "SelfAttention.hxx"
#include <iostream>
#include <iomanip>

static double lossOf(SelfAttention &a, const Matrixd &X)
{
    return a.forward(X).sum();
}

static double gradCheck(const char *name, SelfAttention &attn, const Matrixd &X,
                        const Matrixd &W, void (SelfAttention::*setter)(const Matrixd &),
                        const Matrixd &analytic)
{
    const int d = static_cast<int>(W.rows());
    const double eps = 1e-6;
    Matrixd numeric(d, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
        {
            Matrixd Wp = W; Wp(i, j) += eps; (attn.*setter)(Wp);
            double lp = lossOf(attn, X);
            Matrixd Wm = W; Wm(i, j) -= eps; (attn.*setter)(Wm);
            double lm = lossOf(attn, X);
            numeric(i, j) = (lp - lm) / (2 * eps);
        }
    (attn.*setter)(W); // restore
    double md = (analytic - numeric).cwiseAbs().maxCoeff();
    std::cout << name << " max_abs_diff " << md << "\n";
    return md;
}

int main()
{
    Eigen::setNbThreads(1);
    const int d = 3, seq = 4;

    SelfAttention attn(d);

    Matrixd Wq(d, d), Wk(d, d), Wv(d, d);
    for (int i = 0; i < d; ++i)
        for (int j = 0; j < d; ++j)
        {
            Wq(i, j) = 0.10 * (i + 1) + 0.02 * (j + 1);
            Wk(i, j) = 0.05 * (i + 1) - 0.03 * (j + 1);
            Wv(i, j) = -0.04 * (i + 1) + 0.06 * (j + 1);
        }
    attn.setWq(Wq); attn.setWk(Wk); attn.setWv(Wv);

    Matrixd X(seq, d);
    for (int i = 0; i < seq; ++i)
        for (int j = 0; j < d; ++j)
            X(i, j) = 0.1 * (i + 1) - 0.05 * (j + 1);

    Matrixd out = attn.forward(X);
    Matrixd dOut = Matrixd::Ones(seq, d);
    Matrixd dX = attn.backward(dOut);

    Matrixd aWq = attn.getGradWq();
    Matrixd aWk = attn.getGradWk();
    Matrixd aWv = attn.getGradWv();

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "ATTN_OUT_SUM " << out.sum() << "\n";
    std::cout << "DX_SUM       " << dX.sum() << "\n";

    double m1 = gradCheck("Wq", attn, X, Wq, &SelfAttention::setWq, aWq);
    double m2 = gradCheck("Wk", attn, X, Wk, &SelfAttention::setWk, aWk);
    double m3 = gradCheck("Wv", attn, X, Wv, &SelfAttention::setWv, aWv);

    double worst = std::max(m1, std::max(m2, m3));
    std::cout << "WORST_DIFF   " << worst << "\n";
    std::cout << (worst < 1e-5 ? "GRADCHECK PASS\n" : "GRADCHECK FAIL\n");
    return 0;
}
