#include <math.h>
#include "LossFunctions.h"

double categorical_CE_loss(const Tensor *predictions, const Tensor *targets, const int classes) {
    if (!predictions || !targets) return 0;

    double sum = 0;
    const double *p = (double*)predictions->data;
    const double *t = (double*)targets->data;
    size_t batch_size = predictions->size / classes;

    for (size_t i = 0; i < batch_size; i++) {
        for (int j = 0; j < classes; j++) {
            size_t idx = i * classes + j;
            sum += t[idx] * log(fmax(EPSILON, p[idx]));
        }
    }
    return -sum / batch_size;
}

double sparse_categorical_CE_loss(const Tensor *predictions, const Tensor *targets) {
    return 0;
}

double kullback_leibler_diverggence(const Tensor *predictions, const Tensor *targets) {
    return 0;
}