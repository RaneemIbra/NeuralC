#ifndef NEURALC_POOLING_H
#define NEURALC_POOLING_H

#include "../Layer.h"

typedef enum {
    POOL_MAX,
    POOL_AVERAGE,
    POOL_GLOBAL_AVERAGE
} PoolingType;

typedef struct {
    PoolingType type;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    
    const Tensor *last_input;
    Tensor *max_indices;
} PoolingLayer;

Layer* pooling_layer_create(PoolingType type, int k_h, int k_w, int s_h, int s_w, const char* name);

void free_pooling_layer(PoolingLayer *layer);
Layer* global_average_pooling_layer_create(const char* name);

NcStatus pooling_forward(PoolingLayer *layer, const Tensor *input, Tensor **output, NcGraph *graph);
NcStatus pooling_backward(PoolingLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph);

#endif //NEURALC_POOLING_H
