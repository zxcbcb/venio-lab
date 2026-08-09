#pragma once
#include <Eigen/Core>
#include "Config.hxx"
#include "SelfAttention.hxx"

// Minimal transformer encoder block (forward):
//   r1 = LayerNorm(X + SelfAttention(X))
//   f  = ReLU(r1 W1 + b1) W2 + b2          (position-wise feed-forward)
//   out= LayerNorm(r1 + f)
// LayerNorm here is parameter-free (gamma=1, beta=0). Built on the
// gradient-checked SelfAttention core.
class TransformerBlock
{
private:
    int _d, _hidden;
    SelfAttention _attn;
    Matrixd _W1, _W2;          // d x hidden, hidden x d
    Eigen::RowVectorXd _b1, _b2;

    static Matrixd layerNorm(const Matrixd &X, double eps = 1e-5);
    static Matrixd relu(const Matrixd &X);

public:
    TransformerBlock(int d_model, int hidden);

    Matrixd forward(const Matrixd &X);

    void setW1(const Matrixd &m);
    void setW2(const Matrixd &m);
    void setB1(const Eigen::RowVectorXd &v);
    void setB2(const Eigen::RowVectorXd &v);
    SelfAttention &attention() { return _attn; }
};
