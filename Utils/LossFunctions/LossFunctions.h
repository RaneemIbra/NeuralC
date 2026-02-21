#ifndef NEURALC_LOSSFUNCTIONS_H
#define NEURALC_LOSSFUNCTIONS_H

#define HUBER_LOSS_DELTA 1.345
#define EPSILON 1e-15

typedef double (*LossFunction)(
    const double *predictions,
    const double *targets,
    int length
);

// Regression loss functions
double mse_loss(const double *predictions, const double *targets, int length);

double mae_loss(const double *predictions, const double *targets, int length);

double huber_loss(const double *predictions, const double *targets, int length);

double log_cosh_loss(const double *predictions, const double *targets, int length);

// Classification loss functions
double binary_CE_loss(const double *predictions, const double *targets, int length);

double hinge_loss(const double *predictions, const double *targets, int length);

double squared_hinge_loss(const double *predictions, const double *targets, int length);

// Probabilistic loss functions
double categorical_CE_loss(const double *predictions, const double *targets, int length, int classes);

double sparse_categorical_CE_loss(const double *predictions, const double *targets, int length);

double kullback_leibler_diverggence(const double *predictions, const double *targets, int length);

#endif //NEURALC_LOSSFUNCTIONS_H
