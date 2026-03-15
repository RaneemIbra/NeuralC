#include "SGD.h"
#include <stdlib.h>
#include <string.h>

SGDOptimizer* optimizer_sgd_create(Tensor **params, int count, double lr) {
    SGDOptimizer *opt = (SGDOptimizer*)malloc(sizeof(SGDOptimizer));
    if (!opt) return NULL;
    opt->params = (Tensor**)malloc(sizeof(Tensor*) * count);
    for (int i = 0; i < count; i++) opt->params[i] = params[i];
    opt->count = count;
    opt->learning_rate = lr;
    return opt;
}

NcStatus optimizer_sgd_step(SGDOptimizer *opt) {
    if (!opt) return NC_ERROR_NULL_POINTER;
    for (int i = 0; i < opt->count; i++) {
        Tensor *t = opt->params[i];
        if (t->grad) {
            double *restrict data = (double*)t->data;
            const double *restrict grad = t->grad;
            for (size_t j = 0; j < t->size; j++) data[j] -= opt->learning_rate * grad[j];
        }
    }
    return NC_SUCCESS;
}

void optimizer_sgd_zero_grad(SGDOptimizer *opt) {
    if (!opt) return;
    for (int i = 0; i < opt->count; i++) {
        if (opt->params[i]->grad) {
            memset(opt->params[i]->grad, 0, opt->params[i]->size * sizeof(double));
        }
    }
}

void optimizer_sgd_free(SGDOptimizer *opt) {
    if (!opt) return;
    free(opt->params);
    free(opt);
}
