#include "ConvolutionLayer.hxx"

ConvolutionLayer::ConvolutionLayer(int kernel_rows, int kernel_cols, ActivationFunction *activation_function)
    : Layer(0, activation_function),
      _kh(kernel_rows), _kw(kernel_cols),
      _conv_bias(0.0), _conv_bias_gradient(0.0)
{
    _kernel = RandomGenerator::generateRandomMatrix(-0.1, 0.1, _kh, _kw);
    _kernel_gradient = Matrixd::Zero(_kh, _kw);
}

void ConvolutionLayer::propogateLayer(Matrixd last_layer_output)
{
    _input_cache = last_layer_output;

    const int oh = static_cast<int>(last_layer_output.rows()) - _kh + 1;
    const int ow = static_cast<int>(last_layer_output.cols()) - _kw + 1;

    Matrixd out(oh, ow);
    for (int i = 0; i < oh; ++i)
        for (int j = 0; j < ow; ++j)
            out(i, j) = (last_layer_output.block(i, j, _kh, _kw).array() * _kernel.array()).sum() + _conv_bias;

    _values = out;
    activateLayer(); // _active_values = activation(_values)
}

Matrixd ConvolutionLayer::backwardLayer(const Matrixd &grad_output, const Matrixd &prev_active_values)
{
    (void)prev_active_values; // conv uses its cached input instead

    Matrixd df = _activation_function->toDerivateMatrix(_values); // oh x ow
    Matrixd delta = grad_output.array() * df.array();             // oh x ow

    const int oh = static_cast<int>(delta.rows());
    const int ow = static_cast<int>(delta.cols());

    // Gradient w.r.t. the kernel: dK(a,b) = sum_{i,j} delta(i,j) * input(i+a, j+b)
    _kernel_gradient = Matrixd::Zero(_kh, _kw);
    for (int a = 0; a < _kh; ++a)
        for (int b = 0; b < _kw; ++b)
        {
            double g = 0.0;
            for (int i = 0; i < oh; ++i)
                for (int j = 0; j < ow; ++j)
                    g += delta(i, j) * _input_cache(i + a, j + b);
            _kernel_gradient(a, b) = g;
        }

    _conv_bias_gradient = delta.sum();

    // Gradient w.r.t. the input (scatter-add = full convolution with the kernel).
    Matrixd dinput = Matrixd::Zero(_input_cache.rows(), _input_cache.cols());
    for (int i = 0; i < oh; ++i)
        for (int j = 0; j < ow; ++j)
            for (int a = 0; a < _kh; ++a)
                for (int b = 0; b < _kw; ++b)
                    dinput(i + a, j + b) += delta(i, j) * _kernel(a, b);

    _weights_gradient = _kernel_gradient; // mirror for optimizer compatibility
    return dinput;
}

void ConvolutionLayer::setKernel(const Matrixd &k)
{
    _kernel = k;
    _kh = static_cast<int>(k.rows());
    _kw = static_cast<int>(k.cols());
    _kernel_gradient = Matrixd::Zero(_kh, _kw);
}
void ConvolutionLayer::setConvBias(double b) { _conv_bias = b; }
Matrixd ConvolutionLayer::getKernel() const { return _kernel; }
Matrixd ConvolutionLayer::getKernelGradient() const { return _kernel_gradient; }
double ConvolutionLayer::getConvBias() const { return _conv_bias; }
double ConvolutionLayer::getConvBiasGradient() const { return _conv_bias_gradient; }
