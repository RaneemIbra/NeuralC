#ifndef NEURALC_SGD_H
#define NEURALC_SGD_H

#include "../../../Core/Tensor.h"
#include "../../../Core/Common.h"

typedef struct {
    Tensor **params;
    int count;
    double learning_rate;
} SGDOptimizer;

SGDOptimizer* optimizer_sgd_create(Tensor **params, int count, double lr);
NcStatus optimizer_sgd_step(SGDOptimizer *opt);
void optimizer_sgd_zero_grad(SGDOptimizer *opt);
void optimizer_sgd_free(SGDOptimizer *opt);

#endif //NEURALC_SGD_H
