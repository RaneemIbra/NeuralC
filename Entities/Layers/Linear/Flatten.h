#ifndef NEURALC_FLATTEN_H
#define NEURALC_FLATTEN_H

#include "../Layer.h"

typedef struct {
    int input_shape[TENSOR_MAX_DIMS];
    int input_ndims;
} FlattenLayer;

Layer* flatten_layer_create(const char* name);

NcStatus flatten_forward(FlattenLayer *layer, const Tensor *input, Tensor **output, NcGraph *graph);
NcStatus flatten_backward(FlattenLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph);

#endif //NEURALC_FLATTEN_H
