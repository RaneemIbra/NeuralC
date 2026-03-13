#include "FeedForward.h"
#include "../../../Utils/Initializers/Initializers.h"
#include "../../../Utils/Initializers/BiasInitializers/BiasInitializers.h"
#include "../../../Core/TensorMath.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void init_feed_forward(FeedForward *layer, const int inputs_amount, const int neurons_amount,
                       double (*activation_function)(double), const char *activation_name,
                       const WeightsInitializer weights_initializer) {
    if (!layer) return;

    layer->inputs_amount = inputs_amount;
    layer->neurons_amount = neurons_amount;
    layer->activation_function = activation_function;

    if (inputs_amount < 1 || neurons_amount < 1) {
        layer->weights = NULL;
        layer->biases = NULL;
        layer->outputs = NULL;
        return;
    }

    int weight_shape[] = {neurons_amount, inputs_amount};
    layer->weights = tensor_zeros(DTYPE_FLOAT64, weight_shape, 2, true);
    if (!layer->weights) return;

    if (weights_initializer) {
        weights_initializer(layer->weights, inputs_amount, neurons_amount);
    }

    int bias_shape[] = {neurons_amount, 1};
    layer->biases = tensor_zeros(DTYPE_FLOAT64, bias_shape, 2, true);
    if (!layer->biases) {
        tensor_free(layer->weights);
        return;
    }

    const BiasInitConfig bias_config = bias_init_from_activation(activation_name);
    const BiasInitializer bias_init_func = get_bias_initializer(bias_config.type);
    apply_bias_initializer(bias_init_func, layer->biases, bias_config.param);

    int out_shape[] = {neurons_amount, 1};
    layer->outputs = tensor_zeros(DTYPE_FLOAT64, out_shape, 2, true);
}

void free_feed_forward(FeedForward *layer) {
    if (layer == NULL) return;
    
    if (layer->weights) tensor_free(layer->weights);
    if (layer->biases) tensor_free(layer->biases);
    if (layer->outputs) tensor_free(layer->outputs);

    layer->weights = NULL;
    layer->biases = NULL;
    layer->outputs = NULL;
    layer->inputs_amount = 0;
    layer->neurons_amount = 0;
    layer->activation_function = NULL;
}

Tensor* forward_pass(FeedForward *layer, const Tensor *inputs) {
    if (layer == NULL || layer->weights == NULL || inputs == NULL) return NULL;

    Tensor *product = tensor_matmul(layer->weights, inputs);
    if (!product) return NULL;

    Tensor *z = tensor_add(product, layer->biases);
    tensor_free(product);

    double (*act)(double) = layer->activation_function;
    if (act) {
        double *data = (double*)z->data;
        for (size_t i = 0; i < z->size; i++) {
            data[i] = act(data[i]);
        }
    }

    if (layer->outputs) tensor_free(layer->outputs);
    layer->outputs = z;

    return z;
}
