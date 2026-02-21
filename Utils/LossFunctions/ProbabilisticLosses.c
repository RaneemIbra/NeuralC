#include <math.h>

#include "LossFunctions.h"

double categorical_CE_loss(const double *predictions, const double *targets, const int length, const int classes) {
    if (!predictions || !targets) {
        return 0;
    }
    double sum = 0;
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < classes; j++) {
            sum += predictions[i] * log(targets[j]);
        }
    }
    return -sum;
}

double sparse_categorical_CE_loss(const double *predictions, const double *targets, int length) {

}

double kullback_leibler_diverggence(const double *predictions, const double *targets, int length) {

}