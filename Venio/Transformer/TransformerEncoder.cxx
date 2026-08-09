#include "TransformerEncoder.hxx"
#include "RandomGenerator.hxx"
#include <cmath>

Matrixd LayerNorm::forward(const Matrixd &X)
{
    const int n = static_cast<int>(X.cols());
    Matrixd Y(X.rows(), X.cols());
    _y = Matrixd(X.rows(), X.cols());
    _invstd = Eigen::VectorXd(X.rows());
    for (int i = 0; i < X.rows(); ++i)
    {
        double mean = X.row(i).mean();
        Eigen::RowVectorXd c = X.row(i).array() - mean;
        double var = c.array().square().sum() / n;
        double is = 1.0 / std::sqrt(var + eps);
        _invstd(i) = is;
        Y.row(i) = c * is;
        _y.row(i) = Y.row(i);
    }
    return Y;
}

Matrixd LayerNorm::backward(const Matrixd &dY)
{
    const int n = static_cast<int>(dY.cols());
    Matrixd dX(dY.rows(), dY.cols());
    for (int i = 0; i < dY.rows(); ++i)
    {
        double m1 = dY.row(i).mean();
        double m2 = (dY.row(i).array() * _y.row(i).array()).sum() / n;
        dX.row(i) = (_invstd(i) * (dY.row(i).array() - m1 - _y.row(i).array() * m2)).matrix();
    }
    return dX;
}

TransformerEncoder::TransformerEncoder(int d_model, int hidden, int num_heads)
    : _d(d_model), _hidden(hidden), _heads(num_heads), _mha(d_model, num_heads)
{
    _W1 = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _d, _hidden);
    _W2 = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _hidden, _d);
    _b1 = Eigen::RowVectorXd::Zero(_hidden);
    _b2 = Eigen::RowVectorXd::Zero(_d);
}

Matrixd TransformerEncoder::forward(const Matrixd &X)
{
    _X = X;
    _attnOut = _mha.forward(X);
    _u1 = X + _attnOut;
    _r1 = _ln1.forward(_u1);
    _hpre = (_r1 * _W1).rowwise() + _b1;
    _h = relu(_hpre);
    _f = (_h * _W2).rowwise() + _b2;
    _u2 = _r1 + _f;
    _out = _ln2.forward(_u2);
    return _out;
}

Matrixd TransformerEncoder::backward(const Matrixd &dOut)
{
    Matrixd du2 = _ln2.backward(dOut);
    Matrixd dr1 = du2; // residual path r1 -> u2
    Matrixd df = du2;  // FFN output path

    // FFN backward: f = relu(r1 W1 + b1) W2 + b2
    Matrixd dh = df * _W2.transpose();
    _gW2 = _h.transpose() * df;
    _gb2 = df.colwise().sum();
    Matrixd dhpre = dh.array() * (_hpre.array() > 0.0).cast<double>();
    Matrixd dr1_ffn = dhpre * _W1.transpose();
    _gW1 = _r1.transpose() * dhpre;
    _gb1 = dhpre.colwise().sum();
    dr1 += dr1_ffn;

    // LayerNorm1 backward
    Matrixd du1 = _ln1.backward(dr1);
    Matrixd dX = du1;     // residual path X -> u1
    Matrixd dattn = du1;  // attention path

    Matrixd dX_attn = _mha.backward(dattn);
    dX += dX_attn;
    return dX;
}
