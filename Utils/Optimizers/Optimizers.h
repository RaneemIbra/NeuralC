#ifndef NEURALC_OPTIMIZERS_H
#define NEURALC_OPTIMIZERS_H

#include "../../Core/Tensor.h"
#include "../../Core/Common.h"

typedef struct {
    Tensor **params;
    int count;
    double learning_rate;
} Optimizer;

/**
 * Creates an optimizer for an array of parameter tensors.
 */
Optimizer* optimizer_sgd_create(Tensor **params, int count, double lr);

/**
 * Performs a single SGD update step: w = w - lr * w.grad
 */
NcStatus optimizer_sgd_step(Optimizer *opt);

typedef struct {
    Tensor **params;
    Tensor **m; // First moment
    Tensor **v; // Second moment
    int count;
    double lr;
    double beta1;
    double beta2;
    double epsilon;
    int t; // Timestep
} AdamOptimizer;

AdamOptimizer* optimizer_adam_create(Tensor **params, int count, double lr);
NcStatus optimizer_adam_step(AdamOptimizer *opt);
void optimizer_adam_zero_grad(AdamOptimizer *opt);
void optimizer_adam_free(AdamOptimizer *opt);

/**
 * Zeros all gradients in the parameter list.
 */
void optimizer_zero_grad(Optimizer *opt);

void optimizer_free(Optimizer *opt);

#endif //NEURALC_OPTIMIZERS_H