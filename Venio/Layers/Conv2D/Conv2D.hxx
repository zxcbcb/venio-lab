#pragma once
#include <Eigen/Core>
#include <vector>
#include "Config.hxx"

// General 2D convolution: multi-channel input, multi-filter output, stride, padding.
// Tensor = vector of channels, each an H x W matrix.
// Weights: _W[o*Cin + i] is the (kh x kw) kernel for output-channel o, input-channel i.
// Pure convolution (no activation) so it composes cleanly; loss/activation applied outside.
using Tensor = std::vector<Matrixd>;

class Conv2D
{
private:
    int _cin, _cout, _kh, _kw, _stride, _pad;
    std::vector<Matrixd> _W;   // size cout*cin, each kh x kw
    std::vector<double> _bias; // size cout
    std::vector<Matrixd> _gW;  // gradients, same layout as _W
    std::vector<double> _gb;   // bias gradients
    Tensor _inCache;

    static Matrixd pad2d(const Matrixd &m, int p);

public:
    Conv2D(int cin, int cout, int kh, int kw, int stride = 1, int pad = 0);

    Tensor forward(const Tensor &in);
    Tensor backward(const Tensor &dOut); // returns gradient w.r.t. input

    // accessors (for tests / optimization)
    void setW(int o, int i, const Matrixd &m) { _W[o * _cin + i] = m; }
    Matrixd getW(int o, int i) const { return _W[o * _cin + i]; }
    Matrixd getGradW(int o, int i) const { return _gW[o * _cin + i]; }
    void setBias(int o, double b) { _bias[o] = b; }
    double getGradBias(int o) const { return _gb[o]; }

    int cin() const { return _cin; }
    int cout() const { return _cout; }
    int kh() const { return _kh; }
    int kw() const { return _kw; }
    int outH(int H) const { return (H + 2 * _pad - _kh) / _stride + 1; }
    int outW(int W) const { return (W + 2 * _pad - _kw) / _stride + 1; }
};
