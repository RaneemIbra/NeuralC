#include "Initializers.h"
#include <math.h>
#include <stdlib.h>

double random_uniform(const double min, const double max) {
    const double scale = (double)rand() / (double)RAND_MAX;
    return min + scale * (max - min);
}

void xavier_uniform_init(double *weights, const int input_neurons_amount, const int output_neurons_amount, const int rows, const int cols) {
    const double limit = sqrt(6.0f / (double)(input_neurons_amount + output_neurons_amount));
    const double lower_bound = -limit;
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = random_uniform(lower_bound, limit);
    }
}

void he_uniform_init(double *weights, const int input_neurons_amount, const int output_neurons_amount, const int rows, const int cols) {
    const double limit = sqrt(6.0f / (double)input_neurons_amount);
    const double lower_bound = -limit;
    for (int i = 0; i < rows * cols; i++) {
        weights[i] = random_uniform(lower_bound, limit);
    }
}