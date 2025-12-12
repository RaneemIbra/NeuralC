#ifndef NEURALC_INITIALIZERS_H
#define NEURALC_INITIALIZERS_H

typedef enum {
    INIT_XAVIER_UNIFORM,
    INIT_HE_UNIFORM,
    INIT_LECUN_UNIFORM,

    INIT_XAVIER_NORMAL,
    INIT_HE_NORMAL,
    INIT_LECUN_NORMAL,

    INIT_ORTHOGONAL,
    INIT_SPARSE,
    INIT_RANDOM_NORMAL,
    INIT_RANDOM_UNIFORM
} InitializerType;

typedef struct {
    InitializerType type;
    double param1;
    double param2;
} InitializerConfig;

typedef enum {
    BIAS_ZERO,
    BIAS_CONSTANT,
    BIAS_NORMAL_SMALL
} BiasInitializerType;

typedef void (*BiasInitializer)(double* bias, int count, double param);

void apply_bias_initializer(BiasInitializer init, double* bias, int count, double param);

typedef void (*WeightsInitializer)(double *weights, int fan_in, int fan_out, int rows, int cols);

WeightsInitializer get_initializer(InitializerType type);

void apply_initializer(WeightsInitializer initializer, double *weights, int fan_in, int fan_out, int rows, int cols);

void xavier_uniform_init(double *weights, int fan_in, int fan_out, int rows, int cols);
void he_uniform_init(double *weights, int fan_in, int fan_out, int rows, int cols);
void lecun_uniform_init(double *weights, int fan_in, int fan_out, int rows, int cols);
void xavier_normal_init(double *weights, int fan_in, int fan_out, int rows, int cols);
void he_normal_init(double *weights, int fan_in, int fan_out, int rows, int cols);
void lecun_normal_init(double *weights, int fan_in, int fan_out, int rows, int cols);
void truncated_normal_init(double* w, int rows, int cols, double gain);
void orthogonal_init(double *weights, int fan_in, int fan_out, int rows, int cols);
void sparse_init(double *weights, int fan_in, int fan_out, int rows, int cols);
void constant_init(double* w, int rows, int cols, double gain);
void dirac_init(double* w, int rows, int cols, double gain);
void delta_orthogonal_init(double* w, int rows, int cols, double gain);
void block_diagonal_init(double* w, int rows, int cols, double gain);
double activation_gain(const char* activation_name);
void bias_zero_init(double* b, int count, double param);
void bias_constant_init(double* b, int count, double constant);
void bias_small_normal_init(double* b, int count, double stddev);

#endif
