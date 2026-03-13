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
