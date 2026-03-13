#include <math.h>
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
