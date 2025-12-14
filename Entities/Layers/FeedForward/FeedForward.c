#include "FeedForward.h"
#include "../../../Utils/Initializers/Initializers.h"
#include "../../../Utils/Initializers/BiasInitializers/BiasInitializers.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void init_feed_forward(FeedForward* layer, const int inputs_amount, const int neurons_amount, double (*activation_function)(double), const WeightsInitializer weights_initializer) {
    if (!layer) return;

    if (inputs_amount < 1 || neurons_amount < 1) {
        layer->inputs_amount = 0;
        layer->neurons_amount = 0;
        layer->weights = NULL;
        layer->biases = NULL;
        layer->outputs = NULL;
        layer->activation_function = activation_function;
        return;
    }

    layer->inputs_amount = inputs_amount;
    layer->neurons_amount = neurons_amount;
    layer->activation_function = activation_function;

    const size_t weight_count = (size_t)neurons_amount * (size_t)inputs_amount;
    layer->weights = (double*)malloc(weight_count * sizeof(double));
    if (layer->weights == NULL) {
        layer->biases = NULL;
        layer->outputs = NULL;
        return;
    }
    apply_initializer(
        weights_initializer,
        layer->weights,
        inputs_amount,
        neurons_amount,
        neurons_amount,
        inputs_amount
    );


    layer->biases = (double*)malloc(sizeof(double) * (size_t)neurons_amount);
    if (layer->biases == NULL) {
        free(layer->weights);
        layer->weights = NULL;
        layer->outputs = NULL;
        return;
    }
    apply_bias_initializer(bias_zero_initializer, layer->biases, neurons_amount, 0.0);

    layer->outputs = (double*)malloc(sizeof(double) * (size_t)neurons_amount);
    if (layer->outputs == NULL) {
        free(layer->weights);
        free(layer->biases);
        layer->weights = NULL;
        layer->biases = NULL;
        return;
    }
    memset(layer->outputs, 0, sizeof(double) * (size_t)neurons_amount);
}

void free_feed_forward(FeedForward* layer) {
    if (layer == NULL) return;
    if (layer->weights) {
        free(layer->weights);
        layer->weights = NULL;
    }
    if (layer->biases) {
        free(layer->biases);
        layer->biases = NULL;
    }
    if (layer->outputs) {
        free(layer->outputs);
        layer->outputs = NULL;
    }
    layer->inputs_amount = 0;
    layer->neurons_amount = 0;
    layer->activation_function = NULL;
}

void forward_pass(const FeedForward* layer, const double* inputs) {
    if (layer == NULL || layer->weights == NULL || layer->outputs == NULL) return;
    if (inputs == NULL) return;

    const int in = layer->inputs_amount;
    const int n = layer->neurons_amount;
    double (*act)(double) = layer->activation_function;

    for (int j = 0; j < n; ++j) {
        double sum = layer->biases[j];
        const size_t base = (size_t)j * (size_t)in;
        for (int i = 0; i < in; ++i) {
            sum += layer->weights[base + (size_t)i] * inputs[i];
        }
        layer->outputs[j] = act != NULL ? act(sum) : sum;
    }
}