#include "Initializers.h"
#include <math.h>
#include <string.h>

#include "StructuresInitializers/StructuresInitializers.h"
#include "VarianceInitializers/VarianceInitializers.h"

static void orthogonal_wrapper(Tensor *w, const int fi, const int fo) {
    orthogonal_init((double*)w->data, fi, fo, w->shape[0], w->shape[1], 1.0);
}

static void sparse_wrapper(Tensor *w, const int fi, const int fo) {
    sparse_init((double*)w->data, fi, fo, w->shape[0], w->shape[1], (int) sqrt(fi), 0.01);
}

static void constant_wrapper(Tensor *w, const int fi, const int fo) {
    constant_init((double*)w->data, w->shape[0], w->shape[1], 0.0);
}

static void dirac_wrapper(Tensor *w, const int fi, const int fo) {
    dirac_init((double*)w->data, w->shape[0], w->shape[1], 1.0);
}

static void block_diag_wrapper(Tensor *w, const int fi, const int fo) {
    block_diagonal_init((double*)w->data, w->shape[0], w->shape[1], 1.0);
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

void apply_initializer(const WeightsInitializer init, Tensor *weights, const int fan_in, const int fan_out) {
    if (!init) {
        memset(weights->data, 0, tensor_dtype_size(weights->dtype) * weights->size);
        return;
    }
    init(weights, fan_in, fan_out);
}

double activation_gain(const char *activation_name) {
    if (!activation_name) return 1.0;
    if (!strcmp(activation_name, "relu")) return sqrt(2.0);
    if (!strcmp(activation_name, "leaky_relu")) return sqrt(2.0 / (1 + 0.01 * 0.01));
    if (!strcmp(activation_name, "tanh")) return 5.0 / 3.0;
    return 1.0;
}
