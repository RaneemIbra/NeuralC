#ifndef NEURALC_FEEDFORWARD_H
#define NEURALC_FEEDFORWARD_H

#include "../../../Core/Tensor.h"

typedef void (*WeightsInitializer)(Tensor *weights, int fan_in, int fan_out);

typedef struct {
    int inputs_amount;
    int neurons_amount;
    
    Tensor *weights;
    Tensor *biases;
    Tensor *outputs;

    double (*activation_function)(double);
} FeedForward;

void init_feed_forward(FeedForward *layer, int inputs_amount, int neurons_amount, double (*activation_function)(double),
                       const char *activation_name, WeightsInitializer weights_initializer);

void free_feed_forward(FeedForward *layer);

Tensor* forward_pass(FeedForward *layer, const Tensor *inputs);

#endif //NEURALC_FEEDFORWARD_H
