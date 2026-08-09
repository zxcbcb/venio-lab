#pragma once
#include <Eigen/Core>
#include <vector>
#include "Config.hxx"

// Shared multi-channel tensor type (same alias as Conv2D uses).
using Tensor = std::vector<Matrixd>;

// Max pooling over k x k windows (stride defaults to k). Per channel.
class MaxPool2D
{
private:
    int _k, _stride, _C, _H, _W;
    std::vector<Eigen::MatrixXi> _argY, _argX; // argmax positions per channel

public:
    explicit MaxPool2D(int k, int stride = -1) : _k(k), _stride(stride < 0 ? k : stride),
                                                 _C(0), _H(0), _W(0) {}
    Tensor forward(const Tensor &in);
    Tensor backward(const Tensor &dOut);
};

// Flatten a multi-channel tensor into a 1 x (C*H*W) row (bridges conv -> dense),
// and unflatten on the way back.
class Flatten
{
private:
    int _C, _H, _W;

public:
    Flatten() : _C(0), _H(0), _W(0) {}
    Matrixd forward(const Tensor &in);
    Tensor backward(const Matrixd &dOut);
};
