#include "LinearLayer.h"
#include "../../../Core/Tensor.h"
#include "../../../Utils/Initializers/Initializers.h"
#include "../../../Utils/Initializers/BiasInitializers/BiasInitializers.h"
#include "../../../Core/TensorMath.h"
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* --- Polymorphic Layer Interface Implementation --- */

static NcStatus linear_layer_forward(Layer *self, int num_inputs, const Tensor **inputs, Tensor **output, NcGraph *graph) {
    if (!self || num_inputs != 1 || !inputs[0] || !output) return NC_ERROR_NULL_POINTER;
    return linear_forward_pass((LinearLayer*)self->layer_data, inputs[0], output, graph);
}

static NcStatus linear_layer_backward(Layer *self, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!self || !grad_output || !grad_input) return NC_ERROR_NULL_POINTER;
    return linear_backward_pass((LinearLayer*)self->layer_data, grad_output, grad_input, graph);
}

static Tensor** linear_layer_get_params(Layer *self, int *num_params) {
    if (!self || !num_params) return NULL;
    LinearLayer *ll = (LinearLayer*)self->layer_data;
    
    Tensor **params = (Tensor**)malloc(2 * sizeof(Tensor*));
    params[0] = ll->weights;
    params[1] = ll->biases;
    *num_params = 2;
    return params;
}

static void linear_layer_free_wrapper(Layer *self) {
    if (!self) return;
    if (self->layer_data) {
        free_linear_layer((LinearLayer*)self->layer_data);
        free(self->layer_data);
    }
    if (self->name) free(self->name);
    free(self);
}

Layer* linear_layer_create(int inputs, int neurons, double (*activation)(double), const char* name, WeightsInitializer init) {
    Layer *l = (Layer*)malloc(sizeof(Layer));
    if (!l) return NULL;

    l->name = strdup(name ? name : "linear_dense");
    l->train_mode = true;
    
    LinearLayer *ll = (LinearLayer*)malloc(sizeof(LinearLayer));
    if (!ll) {
        free(l->name);
        free(l);
        return NULL;
    }

    init_linear_layer(ll, inputs, neurons, activation, name, init);
    l->layer_data = ll;

    l->forward = linear_layer_forward;
    l->backward = linear_layer_backward;
    l->get_parameters = linear_layer_get_params;
    l->free = linear_layer_free_wrapper;
    l->reset_state = NULL; 
    l->toggle_train = NULL;
    l->save = NULL;

    return l;
}

/* --- Internal Implementation --- */

void init_linear_layer(LinearLayer *layer, const int inputs_amount, const int neurons_amount,
                       double (*activation_function)(double), const char *activation_name,
                       const WeightsInitializer weights_initializer) {
    if (!layer) return;

    layer->inputs_amount = inputs_amount;
    layer->neurons_amount = neurons_amount;
    layer->activation_function = activation_function;
    layer->last_input = NULL;

    int weight_shape[] = {neurons_amount, inputs_amount};
    layer->weights = tensor_zeros(DTYPE_FLOAT64, weight_shape, 2, true);
    
    if (layer->weights && weights_initializer) {
        weights_initializer(layer->weights, inputs_amount, neurons_amount);
    }

    int bias_shape[] = {neurons_amount, 1};
    layer->biases = tensor_zeros(DTYPE_FLOAT64, bias_shape, 2, true);
    
    if (layer->biases) {
        const BiasInitConfig bias_config = bias_init_from_activation(activation_name);
        const BiasInitializer bias_init_func = get_bias_initializer(bias_config.type);
        apply_bias_initializer(bias_init_func, layer->biases, bias_config.param);
    }

    layer->outputs = NULL;
}

void free_linear_layer(LinearLayer *layer) {
    if (layer == NULL) return;
    if (layer->weights) tensor_free(layer->weights);
    if (layer->biases) tensor_free(layer->biases);
    if (layer->outputs) tensor_free(layer->outputs);
}

NcStatus linear_forward_pass(LinearLayer *layer, const Tensor *inputs, Tensor **output, NcGraph *graph) {
    if (!layer || !layer->weights || !inputs || !output) return NC_ERROR_NULL_POINTER;
    layer->last_input = inputs;

    Tensor *product = NULL;
    NcStatus status = tensor_matmul(layer->weights, inputs, &product);
    if (status != NC_SUCCESS) return status;
    if (graph) nc_graph_track(graph, product);

    Tensor *z = NULL;
    status = tensor_add(product, layer->biases, &z);
    if (!graph) tensor_free(product);
    if (status != NC_SUCCESS) return status;
    if (graph) nc_graph_track(graph, z);

    const double (*act)(double) = layer->activation_function;
    if (act) {
        Tensor *act_out = tensor_create(z->dtype, z->shape, z->ndims, z->requires_grad);
        if (!act_out) return NC_ERROR_OUT_OF_MEMORY;
        if (graph) nc_graph_track(graph, act_out);

        double *restrict src = (double*)z->data;
        double *restrict dst = (double*)act_out->data;
        for (size_t i = 0; i < z->size; i++) dst[i] = act(src[i]);
        z = act_out;
    }

    layer->outputs = z; 
    *output = z;
    return NC_SUCCESS;
}

NcStatus linear_backward_pass(LinearLayer *layer, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph) {
    if (!layer || !grad_output || !grad_input || !layer->last_input) return NC_ERROR_NULL_POINTER;

    // 1. dL/db
    if (layer->biases->grad) {
        printf("    linear_backward: calculating dL/db...\n");
        for (size_t i = 0; i < layer->biases->size; i++) {
            layer->biases->grad[i] += ((double*)grad_output->data)[i];
        }
    }

    // 2. dL/dW
    printf("    linear_backward: calculating dL/dW...\n");
    Tensor *dW = NULL;
    NcStatus status = tensor_matmul_extended(grad_output, false, layer->last_input, true, &dW);
    if (status != NC_SUCCESS) {
        printf("    linear_backward: dL/dW matmul failed with %d\n", status);
        return status;
    }

    if (layer->weights->grad) {
        double *restrict W_g = layer->weights->grad;
        double *restrict dW_d = (double*)dW->data;
        for (size_t i = 0; i < dW->size; i++) W_g[i] += dW_d[i];
    }
    tensor_free(dW);

    // 3. dL/dX
    printf("    linear_backward: calculating dL/dX...\n");
    status = tensor_matmul_extended(layer->weights, true, grad_output, false, grad_input);
    if (status != NC_SUCCESS) {
        printf("    linear_backward: dL/dX matmul failed with %d\n", status);
        return status;
    }
    if (graph) nc_graph_track(graph, *grad_input);

    printf("    linear_backward: completed.\n");
    return NC_SUCCESS;
}
