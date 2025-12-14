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

typedef void (*WeightsInitializer)(double* weights, int fan_in, int fan_out, int rows, int cols);

WeightsInitializer get_initializer(InitializerType type);

void apply_initializer(WeightsInitializer init, double* weights, int fan_in, int fan_out, int rows, int cols);

double activation_gain(const char* activation_name);

#endif /* NEURALC_INITIALIZERS_H */
