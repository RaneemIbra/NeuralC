#include "Graph.h"
#include <stdlib.h>

NcGraph* nc_graph_create() {
    NcGraph *g = (NcGraph*)malloc(sizeof(NcGraph));
    if (!g) return NULL;
    g->capacity = 32;
    g->count = 0;
    g->tensors = (Tensor**)malloc(sizeof(Tensor*) * g->capacity);
    if (!g->tensors) {
        free(g);
        return NULL;
    }
    return g;
}

NcStatus nc_graph_track(NcGraph *g, Tensor *t) {
    if (!g || !t) return NC_ERROR_NULL_POINTER;
    
    if (g->count >= g->capacity) {
        int new_capacity = g->capacity * 2;
        Tensor **new_list = (Tensor**)realloc(g->tensors, sizeof(Tensor*) * new_capacity);
        if (!new_list) return NC_ERROR_OUT_OF_MEMORY;
        g->tensors = new_list;
        g->capacity = new_capacity;
    }
    
    g->tensors[g->count++] = t;
    return NC_SUCCESS;
}

NcStatus nc_graph_untrack(NcGraph *g, const Tensor *t) {
    if (!g || !t) return NC_ERROR_NULL_POINTER;
    for (int i = 0; i < g->count; i++) {
        if (g->tensors[i] == t) {
            g->tensors[i] = NULL;
            return NC_SUCCESS;
        }
    }
    return NC_SUCCESS;
}

void nc_graph_free(NcGraph *g) {
    if (!g) return;
    for (int i = 0; i < g->count; i++) {
        if (g->tensors[i]) {
            tensor_free(g->tensors[i]);
        }
    }
    free(g->tensors);
    free(g);
}
