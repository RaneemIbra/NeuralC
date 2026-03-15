#ifndef NEURALC_MODEL_H
#define NEURALC_MODEL_H

#include "../Layers/Layer.h"

typedef struct {
    Layer **layers;
    int count;
    int capacity;
    NcGraph *session_graph; // Persists intermediate tensors for backprop
} Model;

Model* model_create();
void model_add(Model *m, Layer *l);
NcStatus model_forward(Model *m, const Tensor *input, Tensor **output);
NcStatus model_backward(Model *m, const Tensor *grad_output, NcGraph *graph);
Tensor** model_get_parameters(Model *m, int *count);
void model_free(Model *m);

#endif //NEURALC_MODEL_H
