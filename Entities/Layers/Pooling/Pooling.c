#include "Pooling.h"
#include <stdlib.h>
#include <string.h>
#include <float.h>

/* --- Polymorphic Layer Wrapper --- */

static NcStatus pooling_layer_forward(Layer *self, int num_inputs, const Tensor **inputs, Tensor **output, NcGraph *graph) {
    if (!self || num_inputs != 1) return NC_ERROR_NULL_POINTER;
    return pooling_forward((PoolingLayer*)self->layer_data, inputs[0], output, graph);
}

static NcStatus pooling_layer_backward(Layer *self, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!self) return NC_ERROR_NULL_POINTER;
    return pooling_backward((PoolingLayer*)self->layer_data, grad_output, grad_input, graph);
}

static void pooling_layer_free_wrapper(Layer *self) {
    if (!self) return;
    free_pooling_layer((PoolingLayer*)self->layer_data);
    free(self->layer_data);
    if (self->name) free(self->name);
    free(self);
}

Layer* pooling_layer_create(PoolingType type, int k_h, int k_w, int s_h, int s_w, const char* name) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) return NULL;

    l->name = strdup(name ? name : (type == POOL_MAX ? "max_pool" : "avg_pool"));
    l->train_mode = true;

    PoolingLayer *pl = (PoolingLayer*)malloc(sizeof(PoolingLayer));
    pl->type = type;
    pl->kernel_h = k_h;
    pl->kernel_w = k_w;
    pl->stride_h = s_h;
    pl->stride_w = s_w;
    pl->last_input = NULL;
    pl->max_indices = NULL;

    l->layer_data = pl;
    l->forward = pooling_layer_forward;
    l->backward = pooling_layer_backward;
    l->free = pooling_layer_free_wrapper;
    l->get_parameters = NULL;
    l->reset_state = NULL;
    l->toggle_train = NULL;
    *output = res;
    return NC_SUCCESS;
}

Layer* global_average_pooling_layer_create(const char* name) {
    return pooling_layer_create(POOL_GLOBAL_AVERAGE, 0, 0, 0, 0, name);
}

NcStatus pooling_forward(PoolingLayer *layer, const Tensor *input, Tensor **output, NcGraph *graph) {
    if (!layer || !input || !output) return NC_ERROR_NULL_POINTER;
    layer->last_input = input;

    int C = input->shape[0];
    int H = input->shape[1];
    int W = input->shape[2];

    int out_h, out_w;
    if (layer->type == POOL_GLOBAL_AVERAGE) {
        out_h = 1; out_w = 1;
    } else {
        out_h = (H - layer->kernel_h) / layer->stride_h + 1;
        out_w = (W - layer->kernel_w) / layer->stride_w + 1;
    }

    int out_shape[] = {C, out_h, out_w};
    Tensor *res = tensor_zeros(DTYPE_FLOAT64, out_shape, 3, input->requires_grad);
    if (!res) return NC_ERROR_OUT_OF_MEMORY;
    if (graph) nc_graph_track(graph, res);
    
    // ... rest of logic updated to handle GAP ...

    if (layer->type == POOL_MAX) {
        if (layer->max_indices) tensor_free(layer->max_indices);
        layer->max_indices = tensor_zeros(DTYPE_INT32, out_shape, 3, false);
    }

    double *out_data = (double*)res->data;
    const double *in_data = (const double*)input->data;

    for (int c = 0; c < C; c++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int h_start = oh * layer->stride_h;
                int w_start = ow * layer->stride_w;
                
                if (layer->type == POOL_MAX) {
                    double max_val = -DBL_MAX;
                    int max_idx = -1;
                    for (int kh = 0; kh < layer->kernel_h; kh++) {
                        for (int kw = 0; kw < layer->kernel_w; kw++) {
                            int curr_h = h_start + kh;
                            int curr_w = w_start + kw;
                            int idx = (c * H + curr_h) * W + curr_w;
                            if (in_data[idx] > max_val) {
                                max_val = in_data[idx];
                                max_idx = idx;
                            }
                        }
                    }
                    out_data[(c * out_h + oh) * out_w + ow] = max_val;
                    ((int32_t*)layer->max_indices->data)[(c * out_h + oh) * out_w + ow] = max_idx;
                } else {
                    double sum = 0;
                    for (int kh = 0; kh < layer->kernel_h; kh++) {
                        for (int kw = 0; kw < layer->kernel_w; kw++) {
                            sum += in_data[(c * H + (h_start + kh)) * W + (w_start + kw)];
                        }
                    }
                    out_data[(c * out_h + oh) * out_w + ow] = sum / (layer->kernel_h * layer->kernel_w);
                }
            }
        }
    }

    *output = res;
    return NC_SUCCESS;
}

NcStatus pooling_backward(PoolingLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!layer || !grad_output || !grad_input || !layer->last_input) return NC_ERROR_NULL_POINTER;

    int C = layer->last_input->shape[0];
    int H = layer->last_input->shape[1];
    int W = layer->last_input->shape[2];

    Tensor *dX = tensor_zeros(DTYPE_FLOAT64, layer->last_input->shape, 3, false);
    if (!dX) return NC_ERROR_OUT_OF_MEMORY;
    if (graph) nc_graph_track(graph, dX);

    double *dx_data = (double*)dX->data;
    const double *go_data = (const double*)grad_output->data;

    if (layer->type == POOL_MAX) {
        int32_t *indices = (int32_t*)layer->max_indices->data;
        for (size_t i = 0; i < grad_output->size; i++) {
            dx_data[indices[i]] += go_data[i];
        }
    } else {
        int out_h = grad_output->shape[1];
        int out_w = grad_output->shape[2];
        double scale = 1.0 / (layer->kernel_h * layer->kernel_w);

        for (int c = 0; c < C; c++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int h_start = oh * layer->stride_h;
                    int w_start = ow * layer->stride_w;
                    double grad = go_data[(c * out_h + oh) * out_w + ow] * scale;
                    for (int kh = 0; kh < layer->kernel_h; kh++) {
                        for (int kw = 0; kw < layer->kernel_w; kw++) {
                            dx_data[(c * H + (h_start + kh)) * W + (w_start + kw)] += grad;
                        }
                    }
                }
            }
        }
    }

    *grad_input = dX;
    return NC_SUCCESS;
}
