#ifndef NEURALC_BIASINITIALIZERS_H
#define NEURALC_BIASINITIALIZERS_H

#include "../../../Core/Tensor.h"

typedef void (*BiasInitializer)(Tensor *b, double param);

typedef enum {
    BIAS_INIT_ZERO,
    BIAS_INIT_CONSTANT,
    BIAS_INIT_SMALL_NORMAL
} BiasInitializerType;

typedef struct {
    BiasInitializerType type;
    double param;
} BiasInitConfig;

void bias_zero_initializer(Tensor *b, double unused);

void bias_constant_initializer(Tensor *b, double value);

void bias_small_normal_initializer(Tensor *b, double std);

void apply_bias_initializer(BiasInitializer init, Tensor *bias, double param);

BiasInitializer get_bias_initializer(BiasInitializerType type);

BiasInitConfig bias_init_from_activation(const char *activation_name);

#endif //NEURALC_BIASINITIALIZERS_H
