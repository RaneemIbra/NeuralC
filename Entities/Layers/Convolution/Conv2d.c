#include "Conv2d.h"
#include "../../../Core/TensorMath.h"
#include "../../../Utils/Initializers/BiasInitializers/BiasInitializers.h"
#include <stdlib.h>
#include <string.h>

static NcStatus conv2d_layer_forward(Layer *self, int num_inputs, const Tensor **inputs, Tensor **output, NcGraph *graph) {
    if (!self || num_inputs != 1) return NC_ERROR_NULL_POINTER;
    return conv2d_forward((Conv2dLayer*)self->layer_data, inputs[0], output, graph);
}

static NcStatus conv2d_layer_backward(Layer *self, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!self) return NC_ERROR_NULL_POINTER;
    return conv2d_backward((Conv2dLayer*)self->layer_data, grad_output, grad_input, graph);
}

static Tensor** conv2d_layer_get_params(Layer *self, int *num_params) {
    if (!self) return NULL;
    Conv2dLayer *cl = (Conv2dLayer*)self->layer_data;
    Tensor **params = (Tensor**)malloc(2 * sizeof(Tensor*));
    params[0] = cl->filters;
    params[1] = cl->biases;
    *num_params = 2;
    return params;
}

static void conv2d_layer_free_wrapper(Layer *self) {
    if (!self) return;
    free_conv2d_layer((Conv2dLayer*)self->layer_data);
    free(self->layer_data);
    if (self->name) free(self->name);
    free(self);
}

Layer* conv2d_layer_create(int in_c, int out_c, int k_h, int k_w, int s_h, int s_w, int p_h, int p_w, const char* name, WeightsInitializer init) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) return NULL;

    l->name = strdup(name ? name : "conv2d");
    l->type = NC_LAYER_CONV2D;
    l->train_mode = true;

    Conv2dLayer *cl = (Conv2dLayer*)malloc(sizeof(Conv2dLayer));
    cl->in_channels = in_c;
    cl->out_channels = out_c;
    cl->kernel_h = k_h;
    cl->kernel_w = k_w;
    cl->stride_h = s_h;
    cl->stride_w = s_w;
    cl->pad_h = p_h;
    cl->pad_w = p_w;
    cl->last_input = NULL;

    int filter_shape[] = {out_c, in_c * k_h * k_w};
    cl->filters = tensor_zeros(DTYPE_FLOAT64, filter_shape, 2, true);
    if (init) init(cl->filters, in_c * k_h * k_w, out_c);

    int bias_shape[] = {out_c, 1};
    cl->biases = tensor_zeros(DTYPE_FLOAT64, bias_shape, 2, true);

    l->layer_data = cl;
    l->forward = conv2d_layer_forward;
    l->backward = conv2d_layer_backward;
    l->get_parameters = conv2d_layer_get_params;
    l->free = conv2d_layer_free_wrapper;
    l->reset_state = NULL;
    l->toggle_train = NULL;
    l->save = NULL;

    return l;
}

void free_conv2d_layer(Conv2dLayer *layer) {
    if (!layer) return;
    tensor_free(layer->filters);
    tensor_free(layer->biases);
}

NcStatus conv2d_forward(Conv2dLayer *layer, const Tensor *input, Tensor **output, NcGraph *graph) {
    if (!layer || !input || !output) return NC_ERROR_NULL_POINTER;
    layer->last_input = input;

    int C = input->shape[0];
    int H = input->shape[1];
    int W = input->shape[2];

    Tensor *cols = NULL;
    NcStatus status = nc_im2col(input, C, H, W, layer->kernel_h, layer->kernel_w, 
                                layer->stride_h, layer->stride_w, layer->pad_h, layer->pad_w, &cols);
    if (status != NC_SUCCESS) return status;
    if (graph) nc_graph_track(graph, cols);

    Tensor *res = NULL;
    status = tensor_matmul(layer->filters, cols, &res);
    if (!graph) tensor_free(cols);
    if (status != NC_SUCCESS) return status;
    if (graph) nc_graph_track(graph, res);

    double *r_data = (double*)res->data;
    double *b_data = (double*)layer->biases->data;
    int out_h = (H + 2 * layer->pad_h - layer->kernel_h) / layer->stride_h + 1;
    int out_w = (W + 2 * layer->pad_w - layer->kernel_w) / layer->stride_w + 1;

    for (int oc = 0; oc < layer->out_channels; oc++) {
        for (int i = 0; i < out_h * out_w; i++) {
            r_data[oc * (out_h * out_w) + i] += b_data[oc];
        }
    }

    res->ndims = 3;
    res->shape[0] = layer->out_channels;
    res->shape[1] = out_h;
    res->shape[2] = out_w;

    *output = res;
    return NC_SUCCESS;
}

NcStatus conv2d_backward(Conv2dLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!layer || !grad_output || !grad_input || !layer->last_input) return NC_ERROR_NULL_POINTER;

    int C = layer->last_input->shape[0];
    int H = layer->last_input->shape[1];
    int W = layer->last_input->shape[2];
    int out_h = grad_output->shape[1];
    int out_w = grad_output->shape[2];
    int spatial_size = out_h * out_w;

    if (layer->biases->grad) {
        double *b_g = layer->biases->grad;
        const double *go = (const double*)grad_output->data;
        for (int oc = 0; oc < layer->out_channels; oc++) {
            for (int i = 0; i < spatial_size; i++) {
                b_g[oc] += go[oc * spatial_size + i];
            }
        }
    }

    int original_ndims = grad_output->ndims;
    int original_shape[TENSOR_MAX_DIMS];
    memcpy(original_shape, grad_output->shape, sizeof(int) * TENSOR_MAX_DIMS);
    
    ((Tensor*)grad_output)->ndims = 2;
    ((Tensor*)grad_output)->shape[0] = layer->out_channels;
    ((Tensor*)grad_output)->shape[1] = spatial_size;

    Tensor *cols = NULL;
    NcStatus status = nc_im2col(layer->last_input, C, H, W, layer->kernel_h, layer->kernel_w, 
                                layer->stride_h, layer->stride_w, layer->pad_h, layer->pad_w, &cols);
    if (status != NC_SUCCESS) goto cleanup;

    Tensor *dF = NULL;
    status = tensor_matmul_extended(grad_output, false, cols, true, &dF);
    if (status != NC_SUCCESS) {
        tensor_free(cols);
        goto cleanup;
    }

    if (layer->filters->grad) {
        double *fg = layer->filters->grad;
        double *dfd = (double*)dF->data;
        for (size_t i = 0; i < dF->size; i++) fg[i] += dfd[i];
    }
    tensor_free(dF);

    Tensor *dCols = NULL;
    status = tensor_matmul_extended(layer->filters, true, grad_output, false, &dCols);
    if (status == NC_SUCCESS) {
        status = nc_col2im(dCols, C, H, W, layer->kernel_h, layer->kernel_w, 
                           layer->stride_h, layer->stride_w, layer->pad_h, layer->pad_w, grad_input);
        if (status == NC_SUCCESS && graph) nc_graph_track(graph, *grad_input);
        tensor_free(dCols);
    }
    
    tensor_free(cols);

cleanup:
    ((Tensor*)grad_output)->ndims = original_ndims;
    memcpy(((Tensor*)grad_output)->shape, original_shape, sizeof(int) * TENSOR_MAX_DIMS);
    return status;
}
