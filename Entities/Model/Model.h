#ifndef NEURALC_MODEL_H
#define NEURALC_MODEL_H

#include "../Layers/Layer.h"

typedef struct {
    Layer **layers;
    int count;
    int capacity;
    NcGraph *session_graph;
} NcModel;

NcModel* model_create();
void model_add(NcModel *m, Layer *l);
NcStatus model_forward(NcModel *m, const Tensor *input, Tensor **output);
NcStatus model_backward(NcModel *m, const Tensor *grad_output, NcGraph *graph);
Tensor** model_get_parameters(NcModel *m, int *count);
void model_free(NcModel *m);

#endif //NEURALC_MODEL_H
