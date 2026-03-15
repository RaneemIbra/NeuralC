#ifndef NEURALC_FEEDFORWARD_H
#define NEURALC_FEEDFORWARD_H

#include "../../../Core/Tensor.h"
#include "../Layer.h"

typedef void (*WeightsInitializer)(Tensor *weights, int fan_in, int fan_out);

typedef struct {
    int inputs_amount;
    int neurons_amount;
    
    Tensor *weights;
    Tensor *biases;
    Tensor *outputs;
    const Tensor *last_input; // Tracked for backprop

    double (*activation_function)(double);
} FeedForward;

// Polymorphic Wrapper
Layer* dense_layer_create(int inputs, int neurons, double (*activation)(double), const char* name, WeightsInitializer init);

// Internal implementation details
void init_feed_forward(FeedForward *layer, int inputs_amount, int neurons_amount, double (*activation_function)(double),
                       const char *activation_name, WeightsInitializer weights_initializer);

void free_feed_forward(FeedForward *layer);

NcStatus forward_pass(FeedForward *layer, const Tensor *inputs, Tensor **output, NcGraph *graph);
NcStatus backward_pass(FeedForward *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph);

#endif //NEURALC_FEEDFORWARD_H
