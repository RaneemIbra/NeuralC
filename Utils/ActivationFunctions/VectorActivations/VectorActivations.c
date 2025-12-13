#include "VectorActivations.h"
#include <math.h>

void softmax(const double* input, double* output, const int vector_length) {
    double max = input[0];
    for (int i = 1; i < vector_length; i++)
        if (input[i] > max) max = input[i];

    double sum = 0.0;
    for (int i = 0; i < vector_length; i++) {
        output[i] = exp(input[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < vector_length; i++)
        output[i] /= sum;
}

void log_softmax(const double* input, double* output, const int vector_length) {
    double max = input[0];
    for (int i = 1; i < vector_length; i++)
        if (input[i] > max) max = input[i];

    double sum = 0.0;
    for (int i = 0; i < vector_length; i++)
        sum += exp(input[i] - max);

    const double log_sum = log(sum);

    for (int i = 0; i < vector_length; i++)
        output[i] = input[i] - max - log_sum;
}

void maxout(const double* input, double* output, const int input_size, const int group_size) {
    const int out_size = input_size / group_size;

    for (int i = 0; i < out_size; i++) {
        double max = input[i * group_size];
        for (int j = 1; j < group_size; j++) {
            const double v = input[i * group_size + j];
            if (v > max) max = v;
        }
        output[i] = max;
    }
}
