#include "VectorActivations.h"
#include "../../../Core/Tensor.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void softmax(const double *input, double *output, const int vector_length) {
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

void log_softmax(const double *input, double *output, const int vector_length) {
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

void maxout(const double *input, double *output, const int input_size, const int group_size) {
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

NcStatus softmax_derivative(const Tensor *input, Tensor **jacobian) {
    if (!input || !jacobian) return NC_ERROR_NULL_POINTER;
    
    // For a vector of length N, the Jacobian is NxN
    int N = input->size;
    int res_shape[] = {N, N};
    Tensor *J = tensor_zeros(DTYPE_FLOAT64, res_shape, 2, false);
    if (!J) return NC_ERROR_OUT_OF_MEMORY;

    // First compute softmax
    double *s = (double*)malloc(sizeof(double) * N);
    softmax((const double*)input->data, s, N);

    double *data = (double*)J->data;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            if (i == j) {
                data[i * N + j] = s[i] * (1.0 - s[i]);
            } else {
                data[i * N + j] = -s[i] * s[j];
            }
        }
    }

    free(s);
    *jacobian = J;
    return NC_SUCCESS;
}
