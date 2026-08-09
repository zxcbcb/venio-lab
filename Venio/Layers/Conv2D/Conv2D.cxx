#include "Conv2D.hxx"
#include "RandomGenerator.hxx"

Matrixd Conv2D::pad2d(const Matrixd &m, int p)
{
    if (p == 0)
        return m;
    Matrixd o = Matrixd::Zero(m.rows() + 2 * p, m.cols() + 2 * p);
    o.block(p, p, m.rows(), m.cols()) = m;
    return o;
}

Conv2D::Conv2D(int cin, int cout, int kh, int kw, int stride, int pad)
    : _cin(cin), _cout(cout), _kh(kh), _kw(kw), _stride(stride), _pad(pad)
{
    _W.resize(_cout * _cin);
    for (auto &k : _W)
        k = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _kh, _kw);
    _bias.assign(_cout, 0.0);
    _gW.assign(_cout * _cin, Matrixd::Zero(_kh, _kw));
    _gb.assign(_cout, 0.0);
}

Tensor Conv2D::forward(const Tensor &in)
{
    _inCache = in;
    const int H = static_cast<int>(in[0].rows());
    const int W = static_cast<int>(in[0].cols());
    const int Ho = outH(H), Wo = outW(W);

    std::vector<Matrixd> pin(_cin);
    for (int i = 0; i < _cin; ++i)
        pin[i] = pad2d(in[i], _pad);

    Tensor out(_cout);
    for (int o = 0; o < _cout; ++o)
    {
        Matrixd acc = Matrixd::Constant(Ho, Wo, _bias[o]);
        for (int i = 0; i < _cin; ++i)
        {
            const Matrixd &X = pin[i];
            const Matrixd &K = _W[o * _cin + i];
            for (int y = 0; y < Ho; ++y)
                for (int x = 0; x < Wo; ++x)
                    acc(y, x) += (X.block(y * _stride, x * _stride, _kh, _kw).array() * K.array()).sum();
        }
        out[o] = acc;
    }
    return out;
}

Tensor Conv2D::backward(const Tensor &dOut)
{
    const int H = static_cast<int>(_inCache[0].rows());
    const int W = static_cast<int>(_inCache[0].cols());
    const int Ho = static_cast<int>(dOut[0].rows());
    const int Wo = static_cast<int>(dOut[0].cols());

    std::vector<Matrixd> pin(_cin);
    for (int i = 0; i < _cin; ++i)
        pin[i] = pad2d(_inCache[i], _pad);

    _gW.assign(_cout * _cin, Matrixd::Zero(_kh, _kw));
    _gb.assign(_cout, 0.0);
    std::vector<Matrixd> dpin(_cin, Matrixd::Zero(H + 2 * _pad, W + 2 * _pad));

    for (int o = 0; o < _cout; ++o)
    {
        const Matrixd &d = dOut[o];
        _gb[o] = d.sum();
        for (int i = 0; i < _cin; ++i)
        {
            const Matrixd &X = pin[i];
            const Matrixd &K = _W[o * _cin + i];
            Matrixd &gK = _gW[o * _cin + i];
            Matrixd &dX = dpin[i];
            for (int y = 0; y < Ho; ++y)
                for (int x = 0; x < Wo; ++x)
                {
                    double g = d(y, x);
                    const int ys = y * _stride, xs = x * _stride;
                    gK += g * X.block(ys, xs, _kh, _kw);
                    dX.block(ys, xs, _kh, _kw) += g * K;
                }
        }
    }

    Tensor din(_cin);
    for (int i = 0; i < _cin; ++i)
        din[i] = dpin[i].block(_pad, _pad, H, W);
    return din;
}
