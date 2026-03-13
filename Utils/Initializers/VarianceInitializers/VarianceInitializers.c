#include "VarianceInitializers.h"
#include <math.h>
#include "../RandomInitializers/RandomInitializers.h"

static double fan_value(const int fan_in, const int fan_out, const int mode) {
    if (mode == 0) return fan_in;
    if (mode == 1) return fan_out;
    return 0.5 * (fan_in + fan_out);
}

void xavier_uniform_initializer(Tensor *w, const int fan_in, const int fan_out) {
    const double limit = sqrt(6.0 / (fan_in + fan_out));
    double *data = (double*)w->data;
    for (size_t i = 0; i < w->size; ++i) {
        data[i] = random_uniform_initializer(-limit, limit);
    }
}

void xavier_normal_initializer(Tensor *w, const int fan_in, const int fan_out) {
    const double std = sqrt(2.0 / (fan_in + fan_out));
    double *data = (double*)w->data;
    for (size_t i = 0; i < w->size; ++i) {
        data[i] = random_normal_initializer(0.0, std);
    }
}

void he_uniform_initializer(Tensor *w, const int fan_in, const int fan_out) {
    const double limit = sqrt(6.0 / fan_in);
    double *data = (double*)w->data;
    for (size_t i = 0; i < w->size; ++i) {
        data[i] = random_uniform_initializer(-limit, limit);
    }
}

void he_normal_initializer(Tensor *w, const int fan_in, const int fan_out) {
    const double std = sqrt(2.0 / fan_in);
    double *data = (double*)w->data;
    for (size_t i = 0; i < w->size; ++i) {
        data[i] = random_normal_initializer(0.0, std);
    }
}

void lecun_uniform_initializer(Tensor *w, const int fan_in, const int fan_out) {
    const double limit = sqrt(3.0 / fan_in);
    double *data = (double*)w->data;
    for (size_t i = 0; i < w->size; ++i)
        data[i] = random_uniform_initializer(-limit, limit);
}

void lecun_normal_initializer(Tensor *w, const int fan_in, const int fan_out) {
    const double std = sqrt(1.0 / fan_in);
    double *data = (double*)w->data;
    for (size_t i = 0; i < w->size; ++i)
        data[i] = random_normal_initializer(0.0, std);
}

void variance_scaling(Tensor *w, const int fan_in, const int fan_out,
                      const double scale, const int mode, const int uniform) {
    const double fan = fan_value(fan_in, fan_out, mode);
    const double std = sqrt(scale / fan);
    double *data = (double*)w->data;
    for (size_t i = 0; i < w->size; ++i) {
        data[i] = uniform
                   ? random_uniform_initializer(-sqrt(3.0) * std, sqrt(3.0) * std)
                   : random_normal_initializer(0.0, std);
    }
}

void truncated_normal(Tensor *w, const int fan_in, const int fan_out,
                      const double gain) {
    const double std = gain / sqrt(fan_in);
    double *data = (double*)w->data;
    for (size_t i = 0; i < w->size; ++i)
        data[i] = random_truncated_normal_initializer(std);
}
