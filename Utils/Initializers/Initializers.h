//
// Created by Admin on 12/12/2025.
//

#ifndef NEURALC_INITIALIZERS_H
#define NEURALC_INITIALIZERS_H

void xavier_uniform_init(double *weights, int input_neurons_amount, int output_neurons_amount, int rows, int cols);
void he_uniform_init(double *weights, int input_neurons_amount, int output_neurons_amount ,int rows, int cols);

#endif //NEURALC_INITIALIZERS_H