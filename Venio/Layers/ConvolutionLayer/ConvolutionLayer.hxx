#pragma once
#include "Layer.hxx"

// Single-channel 2D convolution layer (valid cross-correlation, stride 1).
// Input:  H x W matrix
// Kernel: kh x kw learnable weights (+ scalar bias)
// Output: (H-kh+1) x (W-kw+1) feature map, then the activation function.
//
// Forward:   out(i,j) = sum_{a,b} in(i+a, j+b) * K(a,b) + bias
// Backward:  dK(a,b)  = sum_{i,j} delta(i,j) * in(i+a, j+b)
//            dIn      = full-convolution of delta with K (scatter-add)
// where delta = grad_output (.*) activation'(pre-activation).
class ConvolutionLayer : public Layer
{
private:
    int _kh, _kw;                 // kernel height/width
    Matrixd _kernel;              // kh x kw
    Matrixd _kernel_gradient;     // kh x kw
    double  _conv_bias;
    double  _conv_bias_gradient;
    Matrixd _input_cache;         // last input, needed for backward

public:
    ConvolutionLayer(int kernel_rows, int kernel_cols, ActivationFunction *activation_function);

    void propogateLayer(Matrixd last_layer_output) override;
    Matrixd backwardLayer(const Matrixd &grad_output, const Matrixd &prev_active_values) override;

    void setKernel(const Matrixd &k);
    void setConvBias(double b);
    Matrixd getKernel() const;
    Matrixd getKernelGradient() const;
    double  getConvBias() const;
    double  getConvBiasGradient() const;
};
