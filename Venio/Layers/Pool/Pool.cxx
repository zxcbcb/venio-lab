#include "Pool.hxx"

Tensor MaxPool2D::forward(const Tensor &in)
{
    _C = static_cast<int>(in.size());
    _H = static_cast<int>(in[0].rows());
    _W = static_cast<int>(in[0].cols());
    const int Ho = (_H - _k) / _stride + 1;
    const int Wo = (_W - _k) / _stride + 1;

    Tensor out(_C, Matrixd(Ho, Wo));
    _argY.assign(_C, Eigen::MatrixXi(Ho, Wo));
    _argX.assign(_C, Eigen::MatrixXi(Ho, Wo));

    for (int c = 0; c < _C; ++c)
        for (int y = 0; y < Ho; ++y)
            for (int x = 0; x < Wo; ++x)
            {
                int ai = 0, aj = 0;
                double v = in[c].block(y * _stride, x * _stride, _k, _k).maxCoeff(&ai, &aj);
                out[c](y, x) = v;
                _argY[c](y, x) = y * _stride + ai;
                _argX[c](y, x) = x * _stride + aj;
            }
    return out;
}

Tensor MaxPool2D::backward(const Tensor &dOut)
{
    Tensor din(_C, Matrixd::Zero(_H, _W));
    for (int c = 0; c < _C; ++c)
        for (int y = 0; y < dOut[c].rows(); ++y)
            for (int x = 0; x < dOut[c].cols(); ++x)
                din[c](_argY[c](y, x), _argX[c](y, x)) += dOut[c](y, x);
    return din;
}

Matrixd Flatten::forward(const Tensor &in)
{
    _C = static_cast<int>(in.size());
    _H = static_cast<int>(in[0].rows());
    _W = static_cast<int>(in[0].cols());
    Matrixd out(1, _C * _H * _W);
    int idx = 0;
    for (int c = 0; c < _C; ++c)
        for (int y = 0; y < _H; ++y)
            for (int x = 0; x < _W; ++x)
                out(0, idx++) = in[c](y, x);
    return out;
}

Tensor Flatten::backward(const Matrixd &dOut)
{
    Tensor din(_C, Matrixd(_H, _W));
    int idx = 0;
    for (int c = 0; c < _C; ++c)
        for (int y = 0; y < _H; ++y)
            for (int x = 0; x < _W; ++x)
                din[c](y, x) = dOut(0, idx++);
    return din;
}
