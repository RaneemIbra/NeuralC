#ifndef NEURALC_PARAMACTIVATIONS_H
#define NEURALC_PARAMACTIVATIONS_H

#define SELU_ALPHA 1.67326
#define SELU_LAMBDA 1.0507

double elu_activation(double input, double alpha);

double selu_activation(double input);

double prelu_activation(double input, double alpha);

double elu_derivative(double input, double alpha);
double prelu_derivative(double input, double alpha);

#endif //NEURALC_PARAMACTIVATIONS_H
