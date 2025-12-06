#include "ActivationFunctions.h"
#include <math.h>

double relu_activation(const double input) {
    return input > 0 ? input : 0;
}

double leaky_relu_activation(const double input) {
    const double leaky_input = input * 0.01;
    return input > leaky_input ? input : leaky_input;
}

double sigmoid_activation(const double input) {
    return 1 / (1 + exp(-input));
}

double tanH_activation(const double input) {
    return tanh(input);
}

double soft_plus_activation(const double input) {
    return log(1.0 + exp(input));
}

double swish_activation(const double input) {
    return input * sigmoid_activation(input);
}

