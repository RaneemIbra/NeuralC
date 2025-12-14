#include "Initializers.h"
#include <math.h>
#include <string.h>

#include "StructuresInitializers/StructuresInitializers.h"
#include "VarianceInitializers/VarianceInitializers.h"

static void orthogonal_wrapper(double* w, const int fi, const int fo, const int r, const int c) {
    orthogonal_init(w, fi, fo, r, c, 1.0);
}

static void sparse_wrapper(double* w, const int fi, const int fo, const int r, const int c) {
    sparse_init(w, fi, fo, r, c, (int)sqrt(fi), 0.01);
}

static void constant_wrapper(double* w, const int fi, const int fo, const int r, const int c) {
    constant_init(w, r, c, 0.0);
}

static void dirac_wrapper(double* w, const int fi, const int fo, const int r, const int c) {
    dirac_init(w, r, c, 1.0);
}

static void block_diag_wrapper(double* w, const int fi, const int fo, const int r, const int c) {
    block_diagonal_init(w, r, c, 1.0);
}

WeightsInitializer get_initializer(const InitializerType type) {
    switch (type) {
        case INIT_XAVIER_UNIFORM: return xavier_uniform_initializer;
        case INIT_XAVIER_NORMAL: return xavier_normal_initializer;
        case INIT_HE_UNIFORM: return he_uniform_initializer;
        case INIT_HE_NORMAL: return he_normal_initializer;
        case INIT_LECUN_UNIFORM: return lecun_uniform_initializer;
        case INIT_LECUN_NORMAL: return lecun_normal_initializer;

        case INIT_ORTHOGONAL: return orthogonal_wrapper;
        case INIT_SPARSE: return sparse_wrapper;
        case INIT_CONSTANT: return constant_wrapper;
        case INIT_DIRAC: return dirac_wrapper;
        case INIT_BLOCK_DIAGONAL: return block_diag_wrapper;

        default: return NULL;
    }
}

void apply_initializer(const WeightsInitializer init, double* weights, const int fan_in, const int fan_out, const int rows, const int cols) {
    if (!init) {
        memset(weights, 0, sizeof(double) * rows * cols);
        return;
    }
    init(weights, fan_in, fan_out, rows, cols);
}

double activation_gain(const char* activation_name) {
    if (!activation_name) return 1.0;
    if (!strcmp(activation_name, "relu")) return sqrt(2.0);
    if (!strcmp(activation_name, "leaky_relu")) return sqrt(2.0 / (1 + 0.01 * 0.01));
    if (!strcmp(activation_name, "tanh")) return 5.0 / 3.0;
    return 1.0;
}
