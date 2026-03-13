#ifndef NEURALC_LOSSFUNCTIONS_H
#define NEURALC_LOSSFUNCTIONS_H

#include "../../Core/Tensor.h"

#define HUBER_LOSS_DELTA 1.345
#define EPSILON 1e-15

typedef double (*LossFunction)(
    const Tensor *predictions,
    const Tensor *targets
);

// Regression loss functions
double mse_loss(const Tensor *predictions, const Tensor *targets);

double mae_loss(const Tensor *predictions, const Tensor *targets);

double huber_loss(const Tensor *predictions, const Tensor *targets);

double log_cosh_loss(const Tensor *predictions, const Tensor *targets);

// Classification loss functions
double binary_CE_loss(const Tensor *predictions, const Tensor *targets);

double hinge_loss(const Tensor *predictions, const Tensor *targets);

double squared_hinge_loss(const Tensor *predictions, const Tensor *targets);

// Probabilistic loss functions
double categorical_CE_loss(const Tensor *predictions, const Tensor *targets, int classes);

double sparse_categorical_CE_loss(const Tensor *predictions, const Tensor *targets);

double kullback_leibler_diverggence(const Tensor *predictions, const Tensor *targets);

#endif //NEURALC_LOSSFUNCTIONS_H
