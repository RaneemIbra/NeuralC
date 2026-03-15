#ifndef NEURALC_GRAPH_H
#define NEURALC_GRAPH_H

#include "Tensor.h"
#include "Common.h"

/**
 * Graph Collector (Tape)
 * Tracks all intermediate tensors created during a forward pass.
 */
typedef struct {
    Tensor **tensors;
    int count;
    int capacity;
} NcGraph;

NcGraph* nc_graph_create();

/**
 * Adds a tensor to the collector.
 * If the tensor is added, the graph now "owns" it for cleanup.
 */
NcStatus nc_graph_track(NcGraph *g, Tensor *t);

/**
 * Removes a tensor from the collector without freeing it.
 * Used when a tensor needs to persist beyond the graph's lifetime.
 */
NcStatus nc_graph_untrack(NcGraph *g, const Tensor *t);

/**
 * Frees all tensors tracked by the graph and the graph itself.
 * Safe to call even if g is NULL.
 */
void nc_graph_free(NcGraph *g);

#endif //NEURALC_GRAPH_H
