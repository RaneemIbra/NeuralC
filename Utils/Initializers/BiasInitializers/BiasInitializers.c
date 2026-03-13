#include "BiasInitializers.h"
#include "../RandomInitializers/RandomInitializers.h"
#include <string.h>

void bias_zero_initializer(Tensor *b, const double unused) {
    (void) unused;
    memset(b->data, 0, tensor_dtype_size(b->dtype) * b->size);
}

void bias_constant_initializer(Tensor *b, const double value) {
    double *data = (double*)b->data;
    for (size_t i = 0; i < b->size; ++i)
        data[i] = value;
}

void bias_small_normal_initializer(Tensor *b, const double std) {
    double *data = (double*)b->data;
    for (size_t i = 0; i < b->size; ++i)
        data[i] = random_normal_initializer(0.0, std);
}

void apply_bias_initializer(const BiasInitializer init, Tensor *bias, const double param) {
    if (!init)
        bias_zero_initializer(bias, 0.0);
    else
        init(bias, param);
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
