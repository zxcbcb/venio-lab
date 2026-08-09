#pragma once
#include <Eigen/Core>
#include "Config.hxx"

// Single-head scaled dot-product self-attention.
//   Input  X : seq_len x d_model
//   Q=XWq, K=XWk, V=XWv          (each d_model x d_model)
//   S = (Q Kᵀ) / sqrt(d_model)   (seq x seq)
//   A = softmax_rows(S)          (seq x seq)
//   Out = A V                    (seq x d_model)
//
// Backward returns dX and fills the weight gradients (gWq/gWk/gWv), including the
// exact softmax Jacobian per row.
class SelfAttention
{
private:
    int _d;
    Matrixd _Wq, _Wk, _Wv;
    Matrixd _gWq, _gWk, _gWv;
    // caches for backward
    Matrixd _X, _Q, _K, _V, _A;

public:
    explicit SelfAttention(int d_model);

    Matrixd forward(const Matrixd &X);
    Matrixd backward(const Matrixd &dOut); // returns dX

    void setWq(const Matrixd &m);
    void setWk(const Matrixd &m);
    void setWv(const Matrixd &m);
    Matrixd getGradWq() const;
    Matrixd getGradWk() const;
    Matrixd getGradWv() const;

    static Matrixd softmaxRows(const Matrixd &S);
};
