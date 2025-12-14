#include "BiasInitializers.h"
#include "../RandomInitializers/RandomInitializers.h"
#include <string.h>

void bias_zero_initializer(double* b, const int count, const double unused) {
    (void)unused;
    memset(b, 0, sizeof(double) * count);
}

void bias_constant_initializer(double* b, const int count, const double value) {
    for (int i = 0; i < count; ++i)
        b[i] = value;
}

void bias_small_normal_initializer(double* b, const int count, const double std) {
    for (int i = 0; i < count; ++i)
        b[i] = random_normal_initializer(0.0, std);
}

void apply_bias_initializer(const BiasInitializer init, double* bias, const int count, const double param) {
    if (!init)
        bias_zero_initializer(bias, count, 0.0);
    else
        init(bias, count, param);
}