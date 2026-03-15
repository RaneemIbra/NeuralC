#ifndef NEURALC_CONV2D_H
#define NEURALC_CONV2D_H

#include "../Layer.h"
#include "../../../Utils/Initializers/Initializers.h"

typedef struct {
    int in_channels;
    int out_channels;
    int kernel_h;
    int kernel_w;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;

    Tensor *filters;
    Tensor *biases;
    
    const Tensor *last_input;
} Conv2dLayer;

Layer* conv2d_layer_create(int in_c, int out_c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, const char* name, WeightsInitializer init);

void free_conv2d_layer(Conv2dLayer *layer);

NcStatus conv2d_forward(Conv2dLayer *layer, const Tensor *input, Tensor **output, NcGraph *graph);
NcStatus conv2d_backward(Conv2dLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph);

#endif //NEURALC_CONV2D_H
