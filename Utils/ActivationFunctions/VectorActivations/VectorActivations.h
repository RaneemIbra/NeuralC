#ifndef NEURALC_VECTORACTIVATIONS_H
#define NEURALC_VECTORACTIVATIONS_H

void softmax(const double *input, double *output, int vector_length);

void log_softmax(const double *input, double *output, int vector_length);

void maxout(const double *input, double *output, int input_size, int group_size);

/* Derivatives */
// Softmax derivative is a Jacobian, but often simplified in libraries.
// We'll provide a standard Jacobian implementation for general use.
NcStatus softmax_derivative(const Tensor *input, Tensor **jacobian);

#endif //NEURALC_VECTORACTIVATIONS_H
