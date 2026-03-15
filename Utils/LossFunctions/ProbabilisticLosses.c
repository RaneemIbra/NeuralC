#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "LossFunctions.h"

double categorical_CE_loss(const Tensor *predictions, const Tensor *targets, int classes) {
    if (!predictions || !targets) return 0;
    
    const double *p_data = (const double*)predictions->data;
    const double *t_data = (const double*)targets->data;
    
    double total_loss = 0;
    int samples = predictions->size / classes;
    if (samples == 0) samples = 1;

    for (size_t i = 0; i < predictions->size; i++) {
        if (t_data[i] > 0.5) {
            double p = fmax(EPSILON, fmin(1.0 - EPSILON, p_data[i]));
            total_loss += -log(p);
        }
    }
    
    return total_loss / (double)samples;
}

NcStatus categorical_CE_loss_derivative(const Tensor *predictions, const Tensor *targets, Tensor **grad_output) {
    if (!predictions || !targets || !grad_output) return NC_ERROR_NULL_POINTER;
    
    Tensor *grad = tensor_create(predictions->dtype, predictions->shape, predictions->ndims, false);
    if (!grad) return NC_ERROR_OUT_OF_MEMORY;

    const double *P = (const double*)predictions->data;
    const double *T = (const double*)targets->data;
    double *G = (double*)grad->data;
    
    for (size_t i = 0; i < predictions->size; i++) {
        double p = fmax(EPSILON, fmin(1.0 - EPSILON, P[i]));
        G[i] = -T[i] / p;
    }

    *grad_output = grad;
    return NC_SUCCESS;
}

double sparse_categorical_CE_loss(const Tensor *predictions, const Tensor *targets) {
    return 0;
}

double kullback_leibler_diverggence(const Tensor *predictions, const Tensor *targets) {
    if (!predictions || !targets) return 0;
    const double *p = (const double*)predictions->data;
    const double *t = (const double*)targets->data;
    double sum = 0;
    for (size_t i = 0; i < predictions->size; i++) {
        double val_p = fmax(EPSILON, p[i]);
        double val_t = fmax(EPSILON, t[i]);
        sum += val_t * log(val_t / val_p);
    }
    return sum;
}