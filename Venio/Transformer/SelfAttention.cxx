#include "SelfAttention.hxx"
#include "RandomGenerator.hxx"
#include <cmath>

SelfAttention::SelfAttention(int d_model) : _d(d_model)
{
    _Wq = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _d);
    _Wk = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _d);
    _Wv = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _d);
    _gWq = Matrixd::Zero(_d, _d);
    _gWk = Matrixd::Zero(_d, _d);
    _gWv = Matrixd::Zero(_d, _d);
}

Matrixd SelfAttention::softmaxRows(const Matrixd &S)
{
    Matrixd A(S.rows(), S.cols());
    for (int i = 0; i < S.rows(); ++i)
    {
        double m = S.row(i).maxCoeff();
        Eigen::RowVectorXd e = (S.row(i).array() - m).exp();
        A.row(i) = e / e.sum();
    }
    return A;
}

Matrixd SelfAttention::forward(const Matrixd &X)
{
    _X = X;
    _Q = X * _Wq;
    _K = X * _Wk;
    _V = X * _Wv;

    const double scale = 1.0 / std::sqrt(static_cast<double>(_d));
    Matrixd S = (_Q * _K.transpose()) * scale; // seq x seq
    _A = softmaxRows(S);
    return _A * _V; // seq x d
}

Matrixd SelfAttention::backward(const Matrixd &dOut)
{
    const double scale = 1.0 / std::sqrt(static_cast<double>(_d));

    Matrixd dV = _A.transpose() * dOut;   // seq x d
    Matrixd dA = dOut * _V.transpose();   // seq x seq

    // Exact softmax Jacobian per row: dS_i = A_i .* (dA_i - (dA_i . A_i))
    Matrixd dS(dA.rows(), dA.cols());
    for (int i = 0; i < dA.rows(); ++i)
    {
        double dot = (dA.row(i).array() * _A.row(i).array()).sum();
        dS.row(i) = (_A.row(i).array() * (dA.row(i).array() - dot)).matrix();
    }

    Matrixd dQ = (dS * _K) * scale;             // seq x d
    Matrixd dK = (dS.transpose() * _Q) * scale; // seq x d

    _gWq = _X.transpose() * dQ;
    _gWk = _X.transpose() * dK;
    _gWv = _X.transpose() * dV;

    Matrixd dX = dQ * _Wq.transpose() + dK * _Wk.transpose() + dV * _Wv.transpose();
    return dX;
}

void SelfAttention::setWq(const Matrixd &m) { _Wq = m; }
void SelfAttention::setWk(const Matrixd &m) { _Wk = m; }
void SelfAttention::setWv(const Matrixd &m) { _Wv = m; }
Matrixd SelfAttention::getGradWq() const { return _gWq; }
Matrixd SelfAttention::getGradWk() const { return _gWk; }
Matrixd SelfAttention::getGradWv() const { return _gWv; }
