#include "Flatten.h"
#include <stdlib.h>
#include <string.h>

static NcStatus flatten_layer_forward(Layer *self, int num_inputs, const Tensor **inputs, Tensor **output, NcGraph *graph) {
    if (!self || num_inputs != 1) return NC_ERROR_NULL_POINTER;
    return flatten_forward((FlattenLayer*)self->layer_data, inputs[0], output, graph);
}

static NcStatus flatten_layer_backward(Layer *self, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!self) return NC_ERROR_NULL_POINTER;
    return flatten_backward((FlattenLayer*)self->layer_data, grad_output, grad_input, graph);
}

static void flatten_layer_free_wrapper(Layer *self) {
    if (!self) return;
    free(self->layer_data);
    if (self->name) free(self->name);
    free(self);
}

Layer* flatten_layer_create(const char* name) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) return NULL;

    l->name = strdup(name ? name : "flatten");
    l->train_mode = true;

    FlattenLayer *fl = (FlattenLayer*)malloc(sizeof(FlattenLayer));
    l->layer_data = fl;

    l->forward = flatten_layer_forward;
    l->backward = flatten_layer_backward;
    l->free = flatten_layer_free_wrapper;
    l->get_parameters = NULL;
    l->reset_state = NULL;
    l->toggle_train = NULL;
    l->save = NULL;

    return l;
}

NcStatus flatten_forward(FlattenLayer *layer, const Tensor *input, Tensor **output, NcGraph *graph) {
    if (!layer || !input || !output) return NC_ERROR_NULL_POINTER;

    layer->input_ndims = input->ndims;
    for (int i = 0; i < input->ndims; i++) layer->input_shape[i] = input->shape[i];

    int out_shape[] = {(int)input->size, 1};
    Tensor *res = tensor_create(input->dtype, out_shape, 2, input->requires_grad);
    if (!res) return NC_ERROR_OUT_OF_MEMORY;
    if (graph) nc_graph_track(graph, res);

    memcpy(res->data, input->data, input->size * sizeof(double));

    *output = res;
    return NC_SUCCESS;
}

NcStatus flatten_backward(FlattenLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!layer || !grad_output || !grad_input) return NC_ERROR_NULL_POINTER;

    Tensor *dX = tensor_create(DTYPE_FLOAT64, layer->input_shape, layer->input_ndims, false);
    if (!dX) return NC_ERROR_OUT_OF_MEMORY;
    if (graph) nc_graph_track(graph, dX);

    memcpy(dX->data, grad_output->data, grad_output->size * sizeof(double));

    *grad_input = dX;
    return NC_SUCCESS;
}
