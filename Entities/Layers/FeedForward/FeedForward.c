#include "FeedForward.h"
#include "../../../Utils/Initializers/Initializers.h"
#include "../../../Utils/Initializers/BiasInitializers/BiasInitializers.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

void init_feed_forward(FeedForward *layer, const int inputs_amount, const int neurons_amount,
                       double (*activation_function)(double), const char *activation_name,
                       const WeightsInitializer weights_initializer) {
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

    const size_t weight_count = (size_t) neurons_amount * (size_t) inputs_amount;
    const size_t bias_count = (size_t) neurons_amount;
    const size_t output_count = (size_t) neurons_amount;
    const size_t total_count = weight_count + bias_count + output_count;

    double *buffer = (double *) malloc(total_count * sizeof(double));
    if (buffer == NULL) {
        layer->weights = NULL;
        layer->biases = NULL;
        layer->outputs = NULL;
        return;
    }

    layer->weights = buffer;
    layer->biases = buffer + weight_count;
    layer->outputs = buffer + weight_count + bias_count;

    apply_initializer(
        weights_initializer,
        layer->weights,
        inputs_amount,
        neurons_amount,
        neurons_amount,
        inputs_amount
    );

    const BiasInitConfig bias_config = bias_init_from_activation(activation_name);
    const BiasInitializer bias_initializer = get_bias_initializer(bias_config.type);
    apply_bias_initializer(bias_initializer, layer->biases, neurons_amount, 0.0);

    memset(layer->outputs, 0, sizeof(double) * output_count);
}

void free_feed_forward(FeedForward *layer) {
    if (layer == NULL) return;
    
    if (layer->weights) {
        free(layer->weights);
        layer->weights = NULL;
        layer->biases = NULL;
        layer->outputs = NULL;
    }
    
    layer->inputs_amount = 0;
    layer->neurons_amount = 0;
    layer->activation_function = NULL;
}

void forward_pass(const FeedForward *restrict layer, const double *restrict inputs) {
    if (layer == NULL || layer->weights == NULL || layer->outputs == NULL) return;
    if (inputs == NULL) return;

    const int in = layer->inputs_amount;
    const int n = layer->neurons_amount;
    double (*act)(double) = layer->activation_function;

    const double *restrict w = layer->weights;
    const double *restrict b = layer->biases;
    double *restrict out = layer->outputs;

    for (int j = 0; j < n; ++j) {
        double sum = b[j];
        const size_t base = (size_t) j * (size_t) in;
        
        int i = 0;
        int in_limit = in - (in % 4);
        
        for (; i < in_limit; i += 4) {
            sum += w[base + (size_t) i] * inputs[i] +
                   w[base + (size_t)(i + 1)] * inputs[i + 1] +
                   w[base + (size_t)(i + 2)] * inputs[i + 2] +
                   w[base + (size_t)(i + 3)] * inputs[i + 3];
        }
        
        for (; i < in; ++i) {
            sum += w[base + (size_t) i] * inputs[i];
        }
        
        out[j] = act != NULL ? act(sum) : sum;
    }
}
