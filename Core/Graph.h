#ifndef NEURALC_GRAPH_H
#define NEURALC_GRAPH_H

#include "Tensor.h"
#include "Common.h"

typedef struct {
    Tensor **tensors;
    int count;
    int capacity;
} NcGraph;

NcGraph* nc_graph_create();

NcStatus nc_graph_track(NcGraph *g, Tensor *t);

NcStatus nc_graph_untrack(NcGraph *g, const Tensor *t);

void nc_graph_free(NcGraph *g);

#endif //NEURALC_GRAPH_H
