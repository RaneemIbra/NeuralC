#ifndef NEURALC_MODEL_H
#define NEURALC_MODEL_H

#include "../Layers/Layer.h"

typedef struct {
    Layer **layers;
    int count;
    int capacity;
} Model;

Model* model_create();
void model_add(Model *m, Layer *l);
NcStatus model_forward(Model *m, const Tensor *input, Tensor **output);
void model_free(Model *m);

#endif //NEURALC_MODEL_H
