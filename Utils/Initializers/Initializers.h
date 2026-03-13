#ifndef NEURALC_INITIALIZERS_H
#define NEURALC_INITIALIZERS_H

typedef enum {
    INIT_XAVIER_UNIFORM,
    INIT_XAVIER_NORMAL,
    INIT_HE_UNIFORM,
    INIT_HE_NORMAL,
    INIT_LECUN_UNIFORM,
    INIT_LECUN_NORMAL,

    INIT_ORTHOGONAL,
    INIT_SPARSE,
    INIT_CONSTANT,
    INIT_DIRAC,
    INIT_BLOCK_DIAGONAL
} InitializerType;

#include "../../Core/Tensor.h"

typedef void (*WeightsInitializer)(Tensor *weights, int fan_in, int fan_out);

WeightsInitializer get_initializer(InitializerType type);

void apply_initializer(WeightsInitializer init, Tensor *weights, int fan_in, int fan_out);

double activation_gain(const char *activation_name);

#endif /* NEURALC_INITIALIZERS_H */
