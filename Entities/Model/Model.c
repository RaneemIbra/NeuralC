#include "Model.h"
#include "../../Core/Graph.h"
#include <stdlib.h>
#include <stdio.h>

NcModel* model_create() {
    NcModel *m = (NcModel*)malloc(sizeof(NcModel));
    if (!m) return NULL;
    m->capacity = 10;
    m->count = 0;
    m->layers = (Layer**)malloc(sizeof(Layer*) * m->capacity);
    m->session_graph = NULL;
    if (!m->layers) {
        free(m);
        return NULL;
    }
    return m;
}

void model_add(NcModel *m, Layer *l) {
    if (!m || !l) return;
    if (m->count >= m->capacity) {
        m->capacity *= 2;
        m->layers = (Layer**)realloc(m->layers, sizeof(Layer*) * m->capacity);
    }
    m->layers[m->count++] = l;
}

NcStatus model_forward(NcModel *m, const Tensor *input, Tensor **output) {
    if (!m || !output) return NC_ERROR_NULL_POINTER;
    
    if (m->session_graph) nc_graph_free(m->session_graph);
    m->session_graph = nc_graph_create();
    if (!m->session_graph) return NC_ERROR_OUT_OF_MEMORY;

    Tensor *current_input = (Tensor*)input;
    NcStatus status = NC_SUCCESS;

    for (int i = 0; i < m->count; i++) {
        Tensor *layer_output = NULL;
        status = layer_forward_1(m->layers[i], current_input, &layer_output, m->session_graph);
        if (status != NC_SUCCESS) {
            NC_REPORT_ERROR(status);
            return status;
        }
        current_input = layer_output;
    }
    
    *output = current_input;
    return NC_SUCCESS;
}

NcStatus model_backward(NcModel *m, const Tensor *grad_output, NcGraph *graph) {
    if (!m || !grad_output) return NC_ERROR_NULL_POINTER;
    const Tensor *curr_grad = grad_output;
    for (int i = m->count - 1; i >= 0; i--) {
        Tensor *next_grad = NULL;
        NcStatus status = m->layers[i]->backward(m->layers[i], curr_grad, &next_grad, graph);
        if (status != NC_SUCCESS) return status;
        curr_grad = next_grad;
    }
    return NC_SUCCESS;
}

Tensor** model_get_parameters(NcModel *m, int *count) {
    if (!m || !count) return NULL;
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

void model_free(NcModel *m) {
    if (!m) return;
    if (m->session_graph) nc_graph_free(m->session_graph);
    for (int i = 0; i < m->count; i++) {
        if (m->layers[i] && m->layers[i]->free) {
            m->layers[i]->free(m->layers[i]);
        }
    }
    free(m->layers);
    free(m);
}
