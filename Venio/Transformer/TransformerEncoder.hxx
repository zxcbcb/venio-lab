#pragma once
#include <Eigen/Core>
#include "Config.hxx"
#include "MultiHeadAttention.hxx"

// Parameter-free LayerNorm (gamma=1, beta=0) with exact backward.
class LayerNorm
{
public:
    double eps = 1e-5;
    Matrixd _y;                  // normalized output cache
    Eigen::VectorXd _invstd;     // per-row 1/sqrt(var+eps)

    Matrixd forward(const Matrixd &X);
    Matrixd backward(const Matrixd &dY); // dX
};

// Full transformer encoder block with forward AND backward:
//   u1 = X + MHA(X);   r1 = LayerNorm(u1)
//   f  = ReLU(r1 W1 + b1) W2 + b2
//   u2 = r1 + f;       out = LayerNorm(u2)
class TransformerEncoder
{
private:
    int _d, _hidden, _heads;
    MultiHeadAttention _mha;
    LayerNorm _ln1, _ln2;
    Matrixd _W1, _W2;
    Eigen::RowVectorXd _b1, _b2;
    Matrixd _gW1, _gW2;
    Eigen::RowVectorXd _gb1, _gb2;
    // caches
    Matrixd _X, _attnOut, _u1, _r1, _hpre, _h, _f, _u2, _out;

    static Matrixd relu(const Matrixd &X) { return X.cwiseMax(0.0); }

public:
    TransformerEncoder(int d_model, int hidden, int num_heads);

    Matrixd forward(const Matrixd &X);
    Matrixd backward(const Matrixd &dOut); // dX

    MultiHeadAttention &mha() { return _mha; }
    void setW1(const Matrixd &m) { _W1 = m; }
    void setW2(const Matrixd &m) { _W2 = m; }
    void setB1(const Eigen::RowVectorXd &v) { _b1 = v; }
    void setB2(const Eigen::RowVectorXd &v) { _b2 = v; }
    Matrixd getGradW1() const { return _gW1; }
    Matrixd getGradW2() const { return _gW2; }
    Eigen::RowVectorXd getGradB1() const { return _gb1; }
    Eigen::RowVectorXd getGradB2() const { return _gb2; }
};
