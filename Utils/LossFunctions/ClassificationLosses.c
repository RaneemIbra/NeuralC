#include <math.h>
#include <stdlib.h>
#include "LossFunctions.h"

double binary_CE_loss(const Tensor *predictions, const Tensor *targets) {
    if (!predictions || !targets) return 0;

    double sum = 0;
    const double *p_data = (double*)predictions->data;
    const double *t_data = (double*)targets->data;

    for (size_t i = 0; i < predictions->size; i++) {
        const double p = fmax(EPSILON, fmin(1.0 - EPSILON, p_data[i]));
        sum += t_data[i] * log(p) + (1.0 - t_data[i]) * log(1.0 - p);
    }
    return -(sum / predictions->size);
}

NcStatus binary_CE_loss_derivative(const Tensor *predictions, const Tensor *targets, Tensor **grad_output) {
    if (!predictions || !targets || !grad_output) return NC_ERROR_NULL_POINTER;
    
    Tensor *grad = tensor_create(predictions->dtype, predictions->shape, predictions->ndims, false);
    if (!grad) return NC_ERROR_OUT_OF_MEMORY;

    const double *P = (const double*)predictions->data;
    const double *T = (const double*)targets->data;
    double *G = (double*)grad->data;
    const size_t N = predictions->size;

    for (size_t i = 0; i < N; i++) {
        double p = fmax(EPSILON, fmin(1.0 - EPSILON, P[i]));
        G[i] = ((1.0 - T[i]) / (1.0 - p)) - (T[i] / p);
        G[i] /= (double)N;
    }

    *grad_output = grad;
    return NC_SUCCESS;
}

double hinge_loss(const Tensor *predictions, const Tensor *targets) {
    if (!predictions || !targets) return 0;

    double sum = 0;
    const double *p = (double*)predictions->data;
    const double *t = (double*)targets->data;

    for (size_t i = 0; i < predictions->size; i++) {
        sum += fmax(0, 1 - t[i] * p[i]);
    }
    return sum / predictions->size;
}

double squared_hinge_loss(const Tensor *predictions, const Tensor *targets) {
    if (!predictions || !targets) return 0;

    double sum = 0;
    const double *p = (double*)predictions->data;
    const double *t = (double*)targets->data;

    for (size_t i = 0; i < predictions->size; i++) {
        const double value = fmax(0, 1 - t[i] * p[i]);
        sum += value * value;
    }
    return sum / predictions->size;
}
