#pragma once
#include "Layer.hxx"

class SequentialLayer : public Layer
{
public:
    SequentialLayer(int layer_size, ActivationFunction *activation_function)
        : Layer(layer_size, activation_function) {}

    void propogateLayer(Matrixd last_layer_output) override;

    Matrixd backwardLayer(const Matrixd& grad_output, const Matrixd& prev_active_values) override;
};