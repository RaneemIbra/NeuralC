#include "ActivationLayer.h"
#include <stdlib.h>
#include <string.h>

typedef struct {
    double (*activation)(double);
    double (*derivative)(double);
    const Tensor *last_input;
} ActivationData;

static NcStatus activation_forward(Layer *self, int num_inputs, const Tensor **inputs, Tensor **output, NcGraph *graph) {
    if (!self || num_inputs != 1 || !inputs[0] || !output) return NC_ERROR_NULL_POINTER;
    
    const Tensor *restrict in = inputs[0];
    ActivationData *data = (ActivationData*)self->layer_data;
    data->last_input = in;

    Tensor *out = tensor_create(in->dtype, in->shape, in->ndims, in->requires_grad);
    if (!out) return NC_ERROR_OUT_OF_MEMORY;
    if (graph) nc_graph_track(graph, out);

    if (in->dtype == DTYPE_FLOAT64) {
        const double *restrict in_ptr = (const double*)in->data;
        double *restrict out_ptr = (double*)out->data;
        const size_t size = in->size;
        for (size_t i = 0; i < size; i++) {
            out_ptr[i] = data->activation(in_ptr[i]);
        }
    } else {
        if (!graph) tensor_free(out);
        return NC_ERROR_INVALID_DTYPE;
    }

    *output = out;
    return NC_SUCCESS;
}

static NcStatus activation_backward(Layer *self, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!self || !grad_output || !grad_input) return NC_ERROR_NULL_POINTER;
    
    ActivationData *data = (ActivationData*)self->layer_data;
    const Tensor *in = data->last_input;
    if (!in) return NC_ERROR_NULL_POINTER;

    // dL/dX = dL/dY * f'(X)
    Tensor *dX = tensor_create(in->dtype, in->shape, in->ndims, in->requires_grad);
    if (!dX) return NC_ERROR_OUT_OF_MEMORY;
    if (graph) nc_graph_track(graph, dX);

    if (in->dtype == DTYPE_FLOAT64) {
        double *restrict DX = (double*)dX->data;
        const double *restrict DY = (const double*)grad_output->data;
        const double *restrict X = (const double*)in->data;
        const size_t size = in->size;
        for (size_t i = 0; i < size; i++) {
            DX[i] = DY[i] * data->derivative(X[i]);
        }
    } else {
        if (!graph) tensor_free(dX);
        return NC_ERROR_INVALID_DTYPE;
    }

    *grad_input = dX;
    return NC_SUCCESS;
}

static void activation_free(Layer *self) {
    if (!self) return;
    if (self->layer_data) free(self->layer_data);
    if (self->name) free(self->name);
    free(self);
}

Layer* activation_layer_create(double (*activation)(double), double (*derivative)(double), const char* name) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) return NULL;

    l->name = strdup(name ? name : "activation");
    l->train_mode = true;

    ActivationData *data = (ActivationData*)malloc(sizeof(ActivationData));
    if (!data) {
        free(l->name);
        free(l);
        return NULL;
    }
    data->activation = activation;
    data->derivative = derivative;
    data->last_input = NULL;
    l->layer_data = data;

    l->forward = activation_forward;
    l->backward = activation_backward;
    l->free = activation_free;
    l->reset_state = NULL;
    l->toggle_train = NULL;
    l->get_parameters = NULL;
    l->save = NULL;

    return l;
}
