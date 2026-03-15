#ifndef NEURALC_ADAM_H
#define NEURALC_ADAM_H

#include "../../../Core/Tensor.h"
#include "../../../Core/Common.h"

typedef struct {
    Tensor **params;
    Tensor **m;
    Tensor **v;
    int count;
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t;
} AdamOptimizer;

AdamOptimizer* optimizer_adam_create(Tensor **params, int count, double lr);
NcStatus optimizer_adam_step(AdamOptimizer *opt);
void optimizer_adam_zero_grad(AdamOptimizer *opt);
void optimizer_adam_free(AdamOptimizer *opt);

#endif //NEURALC_ADAM_H
