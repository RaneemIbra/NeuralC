#ifndef NEURALC_VECTORACTIVATIONS_H
#define NEURALC_VECTORACTIVATIONS_H

void softmax(const double *input, double *output, int vector_length);

void log_softmax(const double *input, double *output, int vector_length);

void maxout(const double *input, double *output, int input_size, int group_size);

#endif //NEURALC_VECTORACTIVATIONS_H
