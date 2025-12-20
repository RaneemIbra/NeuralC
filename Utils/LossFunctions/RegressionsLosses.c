#include <math.h>

#include "LossFunctions.h"

double mse_loss(const double *predictions, const double *targets, const int length) {
    if (!predictions || !targets) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
        const double diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
    return sum / length;
}

double mae_loss(const double *predictions, const double *targets, const int length) {
    if (!predictions || !targets) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
        const double diff = predictions[i] - targets[i];
        sum += fabs(diff);
    }
    return sum / length;
}

double huber_loss(const double *predictions, const double *targets, const int length) {
    if (!predictions || !targets) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
        const double diff = fabs(predictions[i] - targets[i]);
        if (diff <= HUBER_LOSS_DELTA) {
            sum += 0.5 * diff * diff;
        } else {
            sum += HUBER_LOSS_DELTA * (diff - 0.5 * HUBER_LOSS_DELTA);
        }
    }
    return sum / length;
}

double log_cosh_loss(const double *predictions, const double *targets, const int length) {
    if (!predictions || !targets) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
        const double x = predictions[i] - targets[i];
        const double abs_x = fabs(x);
        sum += abs_x + log1p(exp(-2.0 * abs_x)) - log(2.0);
    }
    return sum / length;
}
