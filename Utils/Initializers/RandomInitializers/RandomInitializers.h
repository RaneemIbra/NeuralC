#ifndef NEURALC_RANDOMINITIALIZERS_H
#define NEURALC_RANDOMINITIALIZERS_H

double random_uniform_initializer(double min, double max);

double random_normal_initializer(double mean, double std);

double random_truncated_normal_initializer(double std);

#endif //NEURALC_RANDOMINITIALIZERS_H
