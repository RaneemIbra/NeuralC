#include <math.h>

#include "LossFunctions.h"

double mse_loss(const Tensor *predictions, const Tensor *targets) {
    if (!predictions || !targets) return 0;

    double sum = 0;
    const double *p = (double*)predictions->data;
    const double *t = (double*)targets->data;

    for (size_t i = 0; i < predictions->size; i++) {
        const double diff = p[i] - t[i];
        sum += diff * diff;
    }
    return sum / predictions->size;
}

double mae_loss(const Tensor *predictions, const Tensor *targets) {
    if (!predictions || !targets) return 0;

    double sum = 0;
    const double *p = (double*)predictions->data;
    const double *t = (double*)targets->data;

    for (size_t i = 0; i < predictions->size; i++) {
        const double diff = p[i] - t[i];
        sum += fabs(diff);
    }
    return sum / predictions->size;
}

double huber_loss(const Tensor *predictions, const Tensor *targets) {
    if (!predictions || !targets) return 0;

    double sum = 0;
    const double *p = (double*)predictions->data;
    const double *t = (double*)targets->data;

    for (size_t i = 0; i < predictions->size; i++) {
        const double diff = fabs(p[i] - t[i]);
        if (diff <= HUBER_LOSS_DELTA) {
            sum += 0.5 * diff * diff;
        } else {
            sum += HUBER_LOSS_DELTA * (diff - 0.5 * HUBER_LOSS_DELTA);
        }
    }
    return sum / predictions->size;
}

double log_cosh_loss(const Tensor *predictions, const Tensor *targets) {
    if (!predictions || !targets) return 0;

    double sum = 0;
    const double *p = (double*)predictions->data;
    const double *t = (double*)targets->data;

    for (size_t i = 0; i < predictions->size; i++) {
        const double x = p[i] - t[i];
        const double abs_x = fabs(x);
        sum += abs_x + log1p(exp(-2.0 * abs_x)) - log(2.0);
    }
    return sum / predictions->size;
}

NcStatus mse_loss_derivative(const Tensor *predictions, const Tensor *targets, Tensor **grad_output) {
    if (!predictions || !targets || !grad_output) return NC_ERROR_NULL_POINTER;
    if (predictions->size != targets->size) return NC_ERROR_SHAPE_MISMATCH;

    Tensor *grad = tensor_create(predictions->dtype, predictions->shape, predictions->ndims, false);
    if (!grad) return NC_ERROR_OUT_OF_MEMORY;

    const double *restrict P = (const double*)predictions->data;
    const double *restrict T = (const double*)targets->data;
    double *restrict G = (double*)grad->data;
    const size_t N = predictions->size;
    const double factor = 2.0 / (double)N;

    for (size_t i = 0; i < N; i++) {
        G[i] = factor * (P[i] - T[i]);
    }

    *grad_output = grad;
    return NC_SUCCESS;
}
