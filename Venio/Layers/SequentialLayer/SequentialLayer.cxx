#include "SequentialLayer.hxx"

void SequentialLayer::propogateLayer(Matrixd last_layer_output)
{
#ifdef CPU_OPTIMIZATION
    _values = (last_layer_output * _weights) + _bias;
#endif
#ifdef GPU_OPTIMIZATION
    _values = K::sumMM(K::multMM(last_layer_output, _weights), _bias);
#endif
    activateLayer();
}
Matrixd SequentialLayer::backwardLayer(const Matrixd& grad_output, const Matrixd& prev_active_values)
{
    Matrixd df = getLayerDerivationMatrix(); // activation derivative at _values
    Matrixd dt, dw, dx;
#ifdef CPU_OPTIMIZATION
    dt = grad_output.array() * df.array();
    dw = prev_active_values.transpose() * dt;
    dx = dt * _weights.transpose();
#endif
#ifdef GPU_OPTIMIZATION
    dt = K::emultMM(grad_output, df);
    dw = K::multMM(K::transposeM(prev_active_values), dt);
    dx = K::multMM(dt, K::transposeM(_weights));
#endif
    _derivation_neurons = dt;
    _weights_gradient = dw;
    _bias_gradient = dt;
    return dx;
}
