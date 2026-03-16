#include "ScalarActivations.h"
#include <math.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double relu_activation(const double input) {
    return input > 0 ? input : 0;
}

double leaky_relu_activation(const double input) {
    const double leaky_input = input * 0.01;
    return input > leaky_input ? input : leaky_input;
}

double sigmoid_activation(const double input) {
    if (input >= 0) {
        return 1 / (1 + exp(-input));
    }
    return exp(input) / (1 + exp(input));
}

double tanh_activation(const double input) {
    return tanh(input);
}

double soft_plus_activation(const double input) {
    if (input > 0) {
        return input + log(1 + exp(-input));
    }
    return log(1.0 + exp(input));
}

double swish_activation(const double input) {
    return input * sigmoid_activation(input);
}

double linear_activation(const double input) {
    return input;
}

double gelu_activation(const double input) {
    return 0.5 * input * (1 + tanh_activation(sqrt(2 / M_PI) * (input + 0.044715 * pow(input, 3))));
}

double hard_sigmoid_activation(const double input) {
    if (input <= -2.5) return 0;
    if (input >= 2.5) return 1;
    return 0.2 * input + 0.5;
}

double hard_tanh_activation(const double input) {
    if (input < -1) return -1;
    if (input > 1) return 1;
    return input;
}

double hard_swish_activation(const double input) {
    return input * hard_sigmoid_activation(input);
}

double mish_activation(const double input) {
    return input * tanh(log(1.0 + exp(input)));
}


double relu_derivative(const double input) {
    return input > 0 ? 1.0 : 0.0;
}

double leaky_relu_derivative(const double input) {
    return input > 0 ? 1.0 : 0.01;
}

double sigmoid_derivative(const double input) {
    double s = sigmoid_activation(input);
    return s * (1.0 - s);
}

double tanh_derivative(const double input) {
    double t = tanh_activation(input);
    return 1.0 - (t * t);
}

double linear_derivative(const double input) {
    return 1.0;
}
