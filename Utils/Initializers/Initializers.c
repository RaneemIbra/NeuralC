#include "Initializers.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static double random_uniform(const double min, const double max) {
    const double scale = (double)rand() / (double)RAND_MAX;
    return min + scale * (max - min);
}

static double random_normal(const double mean, const double stddev) {
    const double u1 = ((double)rand() + 1) / ((double)RAND_MAX + 1);
    const double u2 = ((double)rand() + 1) / ((double)RAND_MAX + 1);
    const double z0 = sqrt(-2.0 * log(u1)) * cos(2 * M_PI * u2);
    return z0 * stddev + mean;
}

static void fill_normal(double *weights, const int size, const double mean, const double stddev) {
    for (int i = 0; i < size; ++i) {
        weights[i] = random_normal(mean, stddev);
    }
}

static void fill_uniform(double *weights, const int size, const double min, const double max) {
    for (int i = 0; i < size; i++) {
        weights[i] = random_uniform(min, max);
    }
}

static double random_truncated_normal(const double stddev) {
    double x;
    do {
        x = random_normal(0.0, stddev);
    } while (fabs(x) > 2.0 * stddev);
    return x;
}

void xavier_uniform_init(double *weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double limit = sqrt(6.0 / (fan_in + fan_out));
    fill_uniform(weights, rows * cols, -limit, limit);
}

void xavier_normal_init(double *weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double stddev = sqrt(2.0 / (fan_in + fan_out));
    fill_normal(weights, rows * cols, 0.0, stddev);
}

void he_uniform_init(double *weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double limit = sqrt(6.0 / fan_in);
    fill_uniform(weights, rows * cols, -limit, limit);
}

void he_normal_init(double *weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double stddev = sqrt(2.0 / fan_in);
    fill_normal(weights, rows * cols, 0.0, stddev);
}

void lecun_uniform_init(double *weights, const int fan_in, int fan_out, const int rows, const int cols) {
    const double limit = sqrt(3.0 / fan_in);
    fill_uniform(weights, rows * cols, -limit, limit);
}

void lecun_normal_init(double *weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    const double stddev = sqrt(1.0 / fan_in);
    fill_normal(weights, rows * cols, 0.0, stddev);
}

void truncated_normal_init(double* w, const int rows, const int cols, const double gain) {
    const double stddev = gain / sqrt(cols);
    for (int i = 0; i < rows * cols; i++)
        w[i] = random_truncated_normal(stddev);
}

void orthogonal_init(double *weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    const int size = rows * cols;
    double *A = malloc(sizeof(double) * size);
    if (!A) return;

    fill_normal(A, size, 0.0, 1.0);

    for (int j = 0; j < cols; j++) {
        double norm = 0.0;
        for (int i = 0; i < rows; i++)
            norm += A[i * cols + j] * A[i * cols + j];
        norm = sqrt(norm);

        if (norm < 1e-8) continue;

        for (int i = 0; i < rows; i++)
            A[i * cols + j] /= norm;

        for (int k = j + 1; k < cols; k++) {
            double dot = 0.0;
            for (int i = 0; i < rows; i++)
                dot += A[i * cols + j] * A[i * cols + k];
            for (int i = 0; i < rows; i++)
                A[i * cols + k] -= dot * A[i * cols + j];
        }
    }

    memcpy(weights, A, sizeof(double) * size);
    free(A);
}

void sparse_init(double *weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    const int nonezeros = (int)sqrt(fan_in);

    memset(weights, 0, sizeof(double) * rows * cols);

    for (int j = 0; j < rows; j++) {
        for (int k = 0; k < nonezeros; k++) {
            const int idx = rand() % cols;
            weights[j * cols + idx] = random_normal(0.0, 0.01);
        }
    }
}

void constant_init(double* w, const int rows, const int cols, const double gain) {
    for (int i = 0; i < rows * cols; i++)
        w[i] = gain;
}

void dirac_init(double* w, const int rows, const int cols, const double gain) {
    memset(w, 0, sizeof(double) * rows * cols);
    const int k = rows < cols ? rows : cols;
    for (int i = 0; i < k; i++)
        w[i * cols + i] = gain;
}

void delta_orthogonal_init(double* w, const int rows, const int cols, const double gain) {
    memset(w, 0, sizeof(double) * rows * cols);

    const int min_dim = rows < cols ? rows : cols;

    double* temp = malloc(sizeof(double) * rows * cols);

    for (int i = 0; i < rows * cols; i++)
        temp[i] = random_normal(0.0, 1.0);

    for (int j = 0; j < min_dim; j++) {
        double norm = 0.0;
        for (int i = 0; i < rows; i++)
            norm += temp[i * cols + j] * temp[i * cols + j];

        norm = sqrt(norm);
        if (norm < 1e-12) continue;

        for (int i = 0; i < rows; i++)
            temp[i * cols + j] /= norm;

        for (int k = j + 1; k < cols; k++) {
            double dot = 0;
            for (int i = 0; i < rows; i++)
                dot += temp[i * cols + j] * temp[i * cols + k];
            for (int i = 0; i < rows; i++)
                temp[i * cols + k] -= dot * temp[i * cols + j];
        }
    }

    for (int i = 0; i < min_dim; i++)
        w[i * cols + i] = temp[i * cols + i] * gain;

    free(temp);
}

void block_diagonal_init(double* w, const int rows, const int cols, const double gain) {
    memset(w, 0, sizeof(double) * rows * cols);

    const int block = rows < cols ? rows : cols;
    for (int i = 0; i < block; i++)
        w[i * cols + i] = gain;
}

double activation_gain(const char* activation_name) {
    if (!activation_name) return 1.0;

    if (strcmp(activation_name, "relu") == 0)
        return sqrt(2.0);

    if (strcmp(activation_name, "leaky_relu") == 0)
        return sqrt(2.0 / (1 + 0.01 * 0.01));

    if (strcmp(activation_name, "tanh") == 0)
        return 5.0 / 3.0;

    if (strcmp(activation_name, "sigmoid") == 0)
        return 1.0;

    if (strcmp(activation_name, "linear") == 0)
        return 1.0;

    return 1.0;
}

WeightsInitializer get_initializer(const InitializerType type) {
    switch (type) {
        case INIT_XAVIER_UNIFORM: return xavier_uniform_init;
        case INIT_HE_UNIFORM: return he_uniform_init;
        case INIT_LECUN_UNIFORM: return lecun_uniform_init;

        case INIT_XAVIER_NORMAL: return xavier_normal_init;
        case INIT_HE_NORMAL: return he_normal_init;
        case INIT_LECUN_NORMAL: return lecun_normal_init;

        case INIT_ORTHOGONAL: return orthogonal_init;
        case INIT_SPARSE: return sparse_init;

        case INIT_RANDOM_NORMAL: return (WeightsInitializer)fill_normal;
        case INIT_RANDOM_UNIFORM: return (WeightsInitializer)fill_uniform;

        default: return NULL;
    }
}

void apply_initializer(const WeightsInitializer initializer, double *weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    if (initializer == NULL) {
        memset(weights, 0, sizeof(double) * rows * cols);
        return;
    }
    initializer(weights, fan_in, fan_out, rows, cols);
}

void bias_zero_init(double* b, const int count, double param) {
    memset(b, 0, count * sizeof(double));
}

void bias_constant_init(double* b, const int count, const double constant) {
    for (int i = 0; i < count; i++)
        b[i] = constant;
}

void bias_small_normal_init(double* b, const int count, const double stddev) {
    for (int i = 0; i < count; i++)
        b[i] = random_normal(0.0, stddev);
}

void apply_bias_initializer(const BiasInitializer init, double* bias, const int count, const double param) {
    if (init == NULL)
        return bias_zero_init(bias, count, 0);

    init(bias, count, param);
}
