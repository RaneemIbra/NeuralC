#ifndef NEURALC_BIASINITIALIZERS_H
#define NEURALC_BIASINITIALIZERS_H

typedef void (*BiasInitializer)(double* b, int count, double param);

void bias_zero_initializer(double* b, int count, double unused);
void bias_constant_initializer(double* b, int count, double value);
void bias_small_normal_initializer(double* b, int count, double std);

void apply_bias_initializer(BiasInitializer init, double* bias, int count, double param);

#endif //NEURALC_BIASINITIALIZERS_H