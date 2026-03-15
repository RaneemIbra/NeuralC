#include "Model.h"
#include "../../Core/Graph.h"
#include <stdlib.h>
#include <stdio.h>

Model* model_create() {
    Model *m = (Model*)malloc(sizeof(Model));
    if (!m) return NULL;
    m->capacity = 10;
    m->count = 0;
    m->layers = (Layer**)malloc(sizeof(Layer*) * m->capacity);
    if (!m->layers) {
        free(m);
        return NULL;
    }
    return m;
}

void model_add(Model *m, Layer *l) {
    if (!m || !l) return;
    if (m->count >= m->capacity) {
        m->capacity *= 2;
        m->layers = (Layer**)realloc(m->layers, sizeof(Layer*) * m->capacity);
    }
    m->layers[m->count++] = l;
}

NcStatus model_forward(Model *m, const Tensor *input, Tensor **output) {
    if (!m || !output) return NC_ERROR_NULL_POINTER;
    if (m->count == 0) {
        *output = (Tensor*)input;
        return NC_SUCCESS;
    }

    NcGraph *session_graph = nc_graph_create();
    if (!session_graph) return NC_ERROR_OUT_OF_MEMORY;

    Tensor *current_input = (Tensor*)input;
    NcStatus status = NC_SUCCESS;

    for (int i = 0; i < m->count; i++) {
        Tensor *layer_output = NULL;
        status = layer_forward_1(m->layers[i], current_input, &layer_output, session_graph);
        if (status != NC_SUCCESS) {
            NC_REPORT_ERROR(status);
            nc_graph_free(session_graph);
            return status;
        }
        current_input = layer_output;
    }

    nc_graph_untrack(session_graph, current_input);
    nc_graph_free(session_graph);
    
    *output = current_input;
    return NC_SUCCESS;
}

NcStatus model_backward(Model *m, const Tensor *grad_output, NcGraph *graph) {
    if (!m || !grad_output) return NC_ERROR_NULL_POINTER;
    
    Tensor *current_grad = (Tensor*)grad_output;
    NcStatus status = NC_SUCCESS;

    for (int i = m->count - 1; i >= 0; i--) {
        Tensor *grad_input = NULL;
        status = m->layers[i]->backward(m->layers[i], current_grad, &grad_input, graph);
        if (status != NC_SUCCESS) {
            NC_REPORT_ERROR(status);
            return status;
        }
        current_grad = grad_input;
    }
    return NC_SUCCESS;
}

Tensor** model_get_parameters(Model *m, int *count) {
    if (!m || !count) return NULL;
    
    // First pass: count all params
    int total = 0;
    for (int i = 0; i < m->count; i++) {
        if (m->layers[i]->get_parameters) {
            int layer_params = 0;
            m->layers[i]->get_parameters(m->layers[i], &layer_params);
            total += layer_params;
        }
    }
    
    Tensor **all_params = (Tensor**)malloc(sizeof(Tensor*) * total);
    int current = 0;
    for (int i = 0; i < m->count; i++) {
        if (m->layers[i]->get_parameters) {
            int layer_count = 0;
            Tensor **layer_params = m->layers[i]->get_parameters(m->layers[i], &layer_count);
            for (int k = 0; k < layer_count; k++) {
                all_params[current++] = layer_params[k];
            }
            free(layer_params);
        }
    }
    *count = total;
    return all_params;
}

void model_free(Model *m) {
    if (!m) return;
    for (int i = 0; i < m->count; i++) {
        if (m->layers[i] && m->layers[i]->free) {
            m->layers[i]->free(m->layers[i]);
        }
    }
    free(m->layers);
    free(m);
}
