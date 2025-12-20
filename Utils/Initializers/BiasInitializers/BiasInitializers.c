#include "BiasInitializers.h"
#include "../RandomInitializers/RandomInitializers.h"
#include <string.h>

void bias_zero_initializer(double *b, const int count, const double unused) {
    (void) unused;
    memset(b, 0, sizeof(double) * count);
}

void bias_constant_initializer(double *b, const int count, const double value) {
    for (int i = 0; i < count; ++i)
        b[i] = value;
}

void bias_small_normal_initializer(double *b, const int count, const double std) {
    for (int i = 0; i < count; ++i)
        b[i] = random_normal_initializer(0.0, std);
}

void apply_bias_initializer(const BiasInitializer init, double *bias, const int count, const double param) {
    if (!init)
        bias_zero_initializer(bias, count, 0.0);
    else
        init(bias, count, param);
}

BiasInitializer get_bias_initializer(const BiasInitializerType type) {
    switch (type) {
        case BIAS_INIT_ZERO: return bias_zero_initializer;
        case BIAS_INIT_CONSTANT: return bias_constant_initializer;
        case BIAS_INIT_SMALL_NORMAL: return bias_small_normal_initializer;
        default: return NULL;
    }
}

BiasInitConfig bias_init_from_activation(const char *activation_name) {
    BiasInitConfig config;
    if (strcmp(activation_name, "relu") == 0 || strcmp(activation_name, "leaky_relu") == 0) {
        config.param = 0.1;
        config.type = BIAS_INIT_CONSTANT;
    } else {
        config.param = 0.1;
        config.type = BIAS_INIT_ZERO;
    }
    return config;
}
