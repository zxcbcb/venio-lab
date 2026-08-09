// Full transformer-encoder-block gradient check: verifies backward through
// LayerNorm + FFN + residuals + multi-head attention. Loss = sum(out), dOut=ones.
#include "TransformerEncoder.hxx"
#include <iostream>
#include <iomanip>
#include <functional>
#include <cmath>

static double lossOf(TransformerEncoder &e, const Matrixd &X) { return e.forward(X).sum(); }

static double gcMat(const char *name, TransformerEncoder &e, const Matrixd &X,
                    const Matrixd &W, std::function<void(const Matrixd &)> setter,
                    const Matrixd &analytic)
{
    const double eps = 1e-6;
    Matrixd numeric(W.rows(), W.cols());
    for (int i = 0; i < W.rows(); ++i)
        for (int j = 0; j < W.cols(); ++j)
        {
            Matrixd Wp = W; Wp(i, j) += eps; setter(Wp);
            double lp = lossOf(e, X);
            Matrixd Wm = W; Wm(i, j) -= eps; setter(Wm);
            double lm = lossOf(e, X);
            numeric(i, j) = (lp - lm) / (2 * eps);
        }
    setter(W);
    double md = (analytic - numeric).cwiseAbs().maxCoeff();
    std::cout << std::left << std::setw(8) << name << std::right
              << " max_abs_diff " << std::scientific << std::setprecision(2) << md << "\n";
    return md;
}

static double gcVec(const char *name, TransformerEncoder &e, const Matrixd &X,
                    const Eigen::RowVectorXd &b, std::function<void(const Eigen::RowVectorXd &)> setter,
                    const Eigen::RowVectorXd &analytic)
{
    const double eps = 1e-6;
    Eigen::RowVectorXd numeric(b.size());
    for (int j = 0; j < b.size(); ++j)
    {
        Eigen::RowVectorXd bp = b; bp(j) += eps; setter(bp);
        double lp = lossOf(e, X);
        Eigen::RowVectorXd bm = b; bm(j) -= eps; setter(bm);
        double lm = lossOf(e, X);
        numeric(j) = (lp - lm) / (2 * eps);
    }
    setter(b);
    double md = (analytic - numeric).cwiseAbs().maxCoeff();
    std::cout << std::left << std::setw(8) << name << std::right
              << " max_abs_diff " << std::scientific << std::setprecision(2) << md << "\n";
    return md;
}

int main()
{
    Eigen::setNbThreads(1);
    const int d = 6, hidden = 8, heads = 2, seq = 4;

    TransformerEncoder enc(d, hidden, heads);

    auto fillM = [](int r, int c, double s) {
        Matrixd m(r, c);
        for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
                m(i, j) = s * std::sin(1.0 * i + 1.7 * j + s);
        return m;
    };
    Matrixd W1 = fillM(d, hidden, 0.10), W2 = fillM(hidden, d, 0.08);
    Eigen::RowVectorXd b1 = Eigen::RowVectorXd::Constant(hidden, 0.01);
    Eigen::RowVectorXd b2 = Eigen::RowVectorXd::Constant(d, -0.02);
    Matrixd Wq = fillM(d, d, 0.11), Wk = fillM(d, d, 0.07), Wv = fillM(d, d, 0.05), Wo = fillM(d, d, 0.09);

    enc.setW1(W1); enc.setW2(W2); enc.setB1(b1); enc.setB2(b2);
    enc.mha().setWq(Wq); enc.mha().setWk(Wk); enc.mha().setWv(Wv); enc.mha().setWo(Wo);

    Matrixd X(seq, d);
    for (int i = 0; i < seq; ++i)
        for (int j = 0; j < d; ++j)
            X(i, j) = 0.1 * (i + 1) - 0.05 * (j + 1);

    Matrixd out = enc.forward(X);
    Matrixd dOut = Matrixd::Ones(seq, d);
    enc.backward(dOut);

    // capture analytic grads (from the single backward at original weights)
    Matrixd aW1 = enc.getGradW1(), aW2 = enc.getGradW2();
    Eigen::RowVectorXd ab1 = enc.getGradB1(), ab2 = enc.getGradB2();
    Matrixd aWq = enc.mha().getGradWq(), aWo = enc.mha().getGradWo();

    std::cout << "ENC_OUT_SHAPE " << out.rows() << "x" << out.cols() << " (heads=" << heads << ", hidden=" << hidden << ")\n";

    double m = 0.0;
    m = std::max(m, gcMat("W1", enc, X, W1, [&](const Matrixd &w){ enc.setW1(w); }, aW1));
    m = std::max(m, gcMat("W2", enc, X, W2, [&](const Matrixd &w){ enc.setW2(w); }, aW2));
    m = std::max(m, gcVec("b1", enc, X, b1, [&](const Eigen::RowVectorXd &v){ enc.setB1(v); }, ab1));
    m = std::max(m, gcVec("b2", enc, X, b2, [&](const Eigen::RowVectorXd &v){ enc.setB2(v); }, ab2));
    m = std::max(m, gcMat("mha.Wq", enc, X, Wq, [&](const Matrixd &w){ enc.mha().setWq(w); }, aWq));
    m = std::max(m, gcMat("mha.Wo", enc, X, Wo, [&](const Matrixd &w){ enc.mha().setWo(w); }, aWo));

    std::cout << "WORST_DIFF " << std::scientific << m << "\n";
    std::cout << (m < 1e-5 ? "ENCODER GRADCHECK PASS\n" : "ENCODER GRADCHECK FAIL\n");
    return 0;
}
