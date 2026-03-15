#ifndef NEURALC_DROPOUT_H
#define NEURALC_DROPOUT_H

#include "../Layer.h"

typedef struct {
    double dropout_rate;
    Tensor *mask;
} DropoutLayer;

Layer* dropout_layer_create(double rate, const char* name);

NcStatus dropout_forward(DropoutLayer *layer, const Tensor *input, Tensor **output, NcGraph *graph, bool train_mode);
NcStatus dropout_backward(DropoutLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph);

#endif //NEURALC_DROPOUT_H
