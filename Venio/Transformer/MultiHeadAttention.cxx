#include "MultiHeadAttention.hxx"
#include "SelfAttention.hxx" // reuse softmaxRows
#include "RandomGenerator.hxx"
#include <cmath>

MultiHeadAttention::MultiHeadAttention(int d_model, int num_heads)
    : _d(d_model), _h(num_heads), _dk(d_model / num_heads)
{
    _Wq = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _d);
    _Wk = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _d);
    _Wv = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _d);
    _Wo = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _d);
}

Matrixd MultiHeadAttention::forward(const Matrixd &X)
{
    _X = X;
    _Q = X * _Wq;
    _K = X * _Wk;
    _V = X * _Wv;

    const int seq = static_cast<int>(X.rows());
    const double scale = 1.0 / std::sqrt(static_cast<double>(_dk));

    _concat = Matrixd(seq, _d);
    _A.assign(_h, Matrixd());

    for (int hd = 0; hd < _h; ++hd)
    {
        Matrixd Qh = _Q.middleCols(hd * _dk, _dk);
        Matrixd Kh = _K.middleCols(hd * _dk, _dk);
        Matrixd Vh = _V.middleCols(hd * _dk, _dk);
        Matrixd S = (Qh * Kh.transpose()) * scale;
        Matrixd A = SelfAttention::softmaxRows(S);
        _A[hd] = A;
        _concat.middleCols(hd * _dk, _dk) = A * Vh;
    }
    return _concat * _Wo;
}

Matrixd MultiHeadAttention::backward(const Matrixd &dOut)
{
    const int seq = static_cast<int>(_X.rows());
    const double scale = 1.0 / std::sqrt(static_cast<double>(_dk));

    _gWo = _concat.transpose() * dOut;
    Matrixd dConcat = dOut * _Wo.transpose(); // seq x d

    Matrixd dQ = Matrixd::Zero(seq, _d);
    Matrixd dK = Matrixd::Zero(seq, _d);
    Matrixd dV = Matrixd::Zero(seq, _d);

    for (int hd = 0; hd < _h; ++hd)
    {
        Matrixd Qh = _Q.middleCols(hd * _dk, _dk);
        Matrixd Kh = _K.middleCols(hd * _dk, _dk);
        Matrixd Vh = _V.middleCols(hd * _dk, _dk);
        const Matrixd &A = _A[hd];
        Matrixd dHead = dConcat.middleCols(hd * _dk, _dk); // seq x dk

        Matrixd dVh = A.transpose() * dHead; // seq x dk
        Matrixd dA = dHead * Vh.transpose(); // seq x seq

        Matrixd dS(seq, seq);
        for (int i = 0; i < seq; ++i)
        {
            double dot = (dA.row(i).array() * A.row(i).array()).sum();
            dS.row(i) = (A.row(i).array() * (dA.row(i).array() - dot)).matrix();
        }

        Matrixd dQh = (dS * Kh) * scale;
        Matrixd dKh = (dS.transpose() * Qh) * scale;

        dQ.middleCols(hd * _dk, _dk) = dQh;
        dK.middleCols(hd * _dk, _dk) = dKh;
        dV.middleCols(hd * _dk, _dk) = dVh;
    }

    _gWq = _X.transpose() * dQ;
    _gWk = _X.transpose() * dK;
    _gWv = _X.transpose() * dV;

    return dQ * _Wq.transpose() + dK * _Wk.transpose() + dV * _Wv.transpose();
}
