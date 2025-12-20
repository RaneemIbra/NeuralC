#include <math.h>
#include "LossFunctions.h"

double binary_CE_loss(const double *predictions, const double *targets, const int length) {
    if (!predictions || !targets) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
        const double p = fmax(EPSILON, fmin(1.0 - EPSILON, predictions[i]));
        sum += targets[i] * log(p) + (1.0 - targets[i]) * log(1.0 - p);
    }
    return -(sum / length);
}

double hinge_loss(const double *predictions, const double *targets, const int length) {
    if (!predictions || !targets) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
        sum += fmax(0, 1 - targets[i] * predictions[i]);
    }
    return sum / length;
}

double squared_hinge_loss(const double *predictions, const double *targets, const int length) {
    if (!predictions || !targets) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
        const double value = fmax(0, 1 - targets[i] * predictions[i]);
        sum += value * value;
    }
    return sum / length;
}
