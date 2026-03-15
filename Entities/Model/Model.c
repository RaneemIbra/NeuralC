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

    // Professional refinement: Use a session graph to track all intermediate allocations
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

    // Protect the final output from being freed by the graph
    nc_graph_untrack(session_graph, current_input);
    
    // Cleanup all intermediate tensors (hidden outputs, temp math products)
    nc_graph_free(session_graph);
    
    *output = current_input;
    return NC_SUCCESS;
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
