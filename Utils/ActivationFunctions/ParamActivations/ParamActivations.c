#include "ParamActivations.h"
#include <math.h>

double elu_activation(const double input, const double alpha) {
    return input > 0 ? input : alpha * (exp(input) - 1);
}

double selu_activation(const double input) {
    return SELU_LAMBDA * elu_activation(input, SELU_ALPHA);
}

double prelu_activation(const double input, const double alpha) {
    return input > 0 ? input : alpha * input;
}
