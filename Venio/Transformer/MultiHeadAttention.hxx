#pragma once
#include <Eigen/Core>
#include <vector>
#include "Config.hxx"

// Multi-head self-attention with output projection.
//   X : seq x d_model, h heads, d_k = d_model / h
//   Q=XWq, K=XWk, V=XWv  (d_model x d_model, all heads at once)
//   per head: softmax(Q_h K_hᵀ / sqrt(d_k)) V_h  -> concat -> * Wo
// Full backward for Wq, Wk, Wv, Wo (exact softmax Jacobian per head).
class MultiHeadAttention
{
private:
    int _d, _h, _dk;
    Matrixd _Wq, _Wk, _Wv, _Wo;
    Matrixd _gWq, _gWk, _gWv, _gWo;
    Matrixd _X, _Q, _K, _V, _concat;
    std::vector<Matrixd> _A; // per-head softmax weights (seq x seq)

public:
    MultiHeadAttention(int d_model, int num_heads);

    Matrixd forward(const Matrixd &X);
    Matrixd backward(const Matrixd &dOut); // returns dX

    void setWq(const Matrixd &m) { _Wq = m; }
    void setWk(const Matrixd &m) { _Wk = m; }
    void setWv(const Matrixd &m) { _Wv = m; }
    void setWo(const Matrixd &m) { _Wo = m; }
    Matrixd getGradWq() const { return _gWq; }
    Matrixd getGradWk() const { return _gWk; }
    Matrixd getGradWv() const { return _gWv; }
    Matrixd getGradWo() const { return _gWo; }
};
