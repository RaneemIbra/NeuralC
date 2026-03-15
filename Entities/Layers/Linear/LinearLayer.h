#ifndef NEURALC_LINEARLAYER_H
#define NEURALC_LINEARLAYER_H

#include "../Layer.h"
#include "../../../Utils/Initializers/Initializers.h"

typedef struct {
    int inputs_amount;
    int neurons_amount;
    
    Tensor *weights;
    Tensor *biases;
    Tensor *outputs;
    const Tensor *last_input;

    double (*activation_function)(double);
} LinearLayer;

Layer* linear_layer_create(int inputs, int neurons, double (*activation)(double), const char* name, WeightsInitializer init);

void init_linear_layer(LinearLayer *layer, int inputs_amount, int neurons_amount, double (*activation_function)(double),
                       const char *activation_name, WeightsInitializer weights_initializer);

void free_linear_layer(LinearLayer *layer);

NcStatus linear_forward_pass(LinearLayer *layer, const Tensor *inputs, Tensor **output, NcGraph *graph);
NcStatus linear_backward_pass(LinearLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph);

#endif //NEURALC_LINEARLAYER_H
