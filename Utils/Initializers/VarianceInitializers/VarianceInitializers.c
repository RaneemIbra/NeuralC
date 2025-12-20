#include "VarianceInitializers.h"
#include <math.h>
#include "../RandomInitializers/RandomInitializers.h"

static double fan_value(const int fan_in, const int fan_out, const int mode) {
    if (mode == 0) return fan_in;
    if (mode == 1) return fan_out;
    return 0.5 * (fan_in + fan_out);
}

void xavier_uniform_initializer(double *w, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double limit = sqrt(6.0 / (fan_in + fan_out));
    for (int i = 0; i < rows * cols; ++i) {
        w[i] = random_uniform_initializer(-limit, limit);
    }
}

void xavier_normal_initializer(double *w, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double std = sqrt(2.0 / (fan_in + fan_out));
    for (int i = 0; i < rows * cols; ++i) {
        w[i] = random_normal_initializer(0.0, std);
    }
}

void he_uniform_initializer(double *w, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double limit = sqrt(6.0 / fan_in);
    for (int i = 0; i < rows * cols; ++i) {
        w[i] = random_uniform_initializer(-limit, limit);
    }
}

void he_normal_initializer(double *w, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double std = sqrt(2.0 / fan_in);
    for (int i = 0; i < rows * cols; ++i) {
        w[i] = random_normal_initializer(0.0, std);
    }
}

void lecun_uniform_initializer(double *w, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double limit = sqrt(3.0 / fan_in);
    for (int i = 0; i < rows * cols; ++i)
        w[i] = random_uniform_initializer(-limit, limit);
}

void lecun_normal_initializer(double *w, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double std = sqrt(1.0 / fan_in);
    for (int i = 0; i < rows * cols; ++i)
        w[i] = random_normal_initializer(0.0, std);
}

void variance_scaling(double *w, const int fan_in, const int fan_out, const int rows, const int cols,
                      const double scale, const int mode, const int uniform) {
    const double fan = fan_value(fan_in, fan_out, mode);
    const double std = sqrt(scale / fan);
    for (int i = 0; i < rows * cols; ++i) {
        w[i] = uniform
                   ? random_uniform_initializer(-sqrt(3.0) * std, sqrt(3.0) * std)
                   : random_normal_initializer(0.0, std);
    }
}

void truncated_normal(double *w, const int fan_in, const int fan_out, const int rows, const int cols,
                      const double gain) {
    const double std = gain / sqrt(fan_in);
    for (int i = 0; i < rows * cols; ++i)
        w[i] = random_truncated_normal_initializer(std);
}
