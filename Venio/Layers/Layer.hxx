#pragma once

#include "ActivationFunction.hxx"
#include "RandomGenerator.hxx"
#include "ErrorLogger.hxx"
#include "Config.hxx"

class Layer
{
protected:
    Matrixd _bias, _bias_gradient;
    Matrixd _values;
    Matrixd _active_values;
    Matrixd _derivation_neurons;
    Matrixd _weights, _weights_gradient;
    ActivationFunction *_activation_function;
    int _layer_size;

public:
    Layer(int layer_size, ActivationFunction *activation_function);
    virtual ~Layer() = default;
    void buildLayer(int output_size_of_last_layer);
    void buildFirstLayer();
    void activateLayer();

    virtual void propogateLayer(Matrixd last_layer_output) = 0;
    // Compute this layer's weight/bias gradients from the gradient flowing in from
    // the next layer (grad_output) and the previous layer's activations, store them,
    // and return the gradient to pass further back. Each layer type defines its own.
    virtual Matrixd backwardLayer(const Matrixd& grad_output, const Matrixd& prev_active_values) = 0;

    void setLayerDerivation(Matrixd new_derivation_neurons_matrix);
    void setLayerBias(Matrixd new_bias_matrix);
    void setLayerBiasGradient(Matrixd new_bias_gradient_matrix);
    void setLayerValues(Matrixd new_values_matrix);
    void setLayerActiveValues(Matrixd new_active_values_matrix);
    void setLayerWeights(Matrixd new_weights_matrix);
    void setLayerWeightsGradient(Matrixd new_weights_gradient_matrix);
    void setLayerActivationFunction(ActivationFunction *new_activation_function);

    Matrixd getLayerDerivationMatrix();

    Matrixd getLayerBias();
    Matrixd getLayerValues();
    Matrixd getLayerActiveValues();
    Matrixd getLayerWeights();
    Matrixd getLayerBiasGradient();
    Matrixd getLayerWeightsGradient();
    Matrixd getLayerDerivation();
    ActivationFunction *getLayerActivationFunction();
    size_t getLayerSize() const;
};