#ifndef NEURALC_BIASINITIALIZERS_H
#define NEURALC_BIASINITIALIZERS_H

typedef void (*BiasInitializer)(double *b, int count, double param);

typedef enum {
    BIAS_INIT_ZERO,
    BIAS_INIT_CONSTANT,
    BIAS_INIT_SMALL_NORMAL
} BiasInitializerType;

typedef struct {
    BiasInitializerType type;
    double param;
} BiasInitConfig;

void bias_zero_initializer(double *b, int count, double unused);

void bias_constant_initializer(double *b, int count, double value);

void bias_small_normal_initializer(double *b, int count, double std);

void apply_bias_initializer(BiasInitializer init, double *bias, int count, double param);

BiasInitializer get_bias_initializer(BiasInitializerType type);

BiasInitConfig bias_init_from_activation(const char *activation_name);

#endif //NEURALC_BIASINITIALIZERS_H
