#ifndef NEURALC_SCALARACTIVATIONS_H
#define NEURALC_SCALARACTIVATIONS_H

double relu_activation(double input);
double leaky_relu_activation(double input);
double sigmoid_activation(double input);
double tanh_activation(double input);
double soft_plus_activation(double input);
double swish_activation(double input);
double linear_activation(double input);
double gelu_activation(double input);

double hard_sigmoid_activation(double input);
double hard_tanh_activation(double input);
double hard_swish_activation(double input);

double mish_activation(double input);

#endif //NEURALC_SCALARACTIVATIONS_H