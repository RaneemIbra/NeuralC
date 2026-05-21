#include "Dropout.h"
#include <stdlib.h>
#include <string.h>

static NcStatus dropout_layer_forward(Layer *self, int num_inputs, const Tensor **inputs, Tensor **output, NcGraph *graph) {
    if (!self || num_inputs != 1) return NC_ERROR_NULL_POINTER;
    return dropout_forward((DropoutLayer*)self->layer_data, inputs[0], output, graph, self->train_mode);
}

static NcStatus dropout_layer_backward(Layer *self, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!self) return NC_ERROR_NULL_POINTER;
    return dropout_backward((DropoutLayer*)self->layer_data, grad_output, grad_input, graph);
}

static void dropout_layer_free_wrapper(Layer *self) {
    if (!self) return;
    DropoutLayer *dl = (DropoutLayer*)self->layer_data;
    if (dl->mask) tensor_free(dl->mask);
    free(dl);
    if (self->name) free(self->name);
    free(self);
}

Layer* dropout_layer_create(double rate, const char* name) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) return NULL;
    l->name = strdup(name ? name : "dropout");
    l->train_mode = true;
    
    DropoutLayer *dl = (DropoutLayer*)malloc(sizeof(DropoutLayer));
    dl->dropout_rate = rate;
    dl->mask = NULL;
    l->layer_data = dl;

    l->forward = dropout_layer_forward;
    l->backward = dropout_layer_backward;
    l->free = dropout_layer_free_wrapper;
    l->get_parameters = NULL;
    l->toggle_train = NULL;
    l->reset_state = NULL;
    l->save = NULL;
    return l;
}

NcStatus dropout_forward(DropoutLayer *layer, const Tensor *input, Tensor **output, NcGraph *graph, bool train_mode) {
    if (!layer || !input || !output) return NC_ERROR_NULL_POINTER;

    if (!train_mode) {
        *output = (Tensor*)input;
        return NC_SUCCESS;
    }

    if (layer->mask) tensor_free(layer->mask);
    layer->mask = tensor_create(input->dtype, input->shape, input->ndims, false);
    
    Tensor *res = tensor_create(input->dtype, input->shape, input->ndims, input->requires_grad);
    if (graph) nc_graph_track(graph, res);

    double scale = 1.0 / (1.0 - layer->dropout_rate);
    double *in_data = (double*)input->data;
    double *mask_data = (double*)layer->mask->data;
    double *res_data = (double*)res->data;

    for (size_t i = 0; i < input->size; i++) {
        double r = (double)rand() / (double)RAND_MAX;
        if (r > layer->dropout_rate) {
            mask_data[i] = scale;
            res_data[i] = in_data[i] * scale;
        } else {
            mask_data[i] = 0;
            res_data[i] = 0;
        }
    }

    *output = res;
    return NC_SUCCESS;
}

NcStatus dropout_backward(DropoutLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!layer || !grad_output || !grad_input || !layer->mask) return NC_ERROR_NULL_POINTER;

    Tensor *dX = tensor_create(grad_output->dtype, grad_output->shape, grad_output->ndims, false);
    if (graph) nc_graph_track(graph, dX);

    double *go = (double*)grad_output->data;
    double *m = (double*)layer->mask->data;
    double *dx_data = (double*)dX->data;

    for (size_t i = 0; i < grad_output->size; i++) {
        dx_data[i] = go[i] * m[i];
    }

    *grad_input = dX;
    return NC_SUCCESS;
}
