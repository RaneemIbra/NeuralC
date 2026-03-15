#include "Optimizers.h"
#include <stdlib.h>

Optimizer* optimizer_sgd_create(Tensor **params, int count, double lr) {
    Optimizer *opt = (Optimizer*)malloc(sizeof(Optimizer));
    if (!opt) return NULL;
    
    opt->params = (Tensor**)malloc(sizeof(Tensor*) * count);
    if (!opt->params) {
        free(opt);
        return NULL;
    }
    
    for (int i = 0; i < count; i++) {
        opt->params[i] = params[i];
    }
    opt->count = count;
    opt->learning_rate = lr;
    
    return opt;
}

NcStatus optimizer_sgd_step(Optimizer *opt) {
    if (!opt) return NC_ERROR_NULL_POINTER;
    
    for (int i = 0; i < opt->count; i++) {
        Tensor *t = opt->params[i];
        if (t->grad) {
            double *restrict data = (double*)t->data;
            const double *restrict grad = t->grad;
            const double lr = opt->learning_rate;
            const size_t size = t->size;
            
            for (size_t j = 0; j < size; j++) {
                data[j] -= lr * grad[j];
            }
        }
    }
    return NC_SUCCESS;
}

void optimizer_zero_grad(Optimizer *opt) {
    if (!opt) return;
    for (int i = 0; i < opt->count; i++) {
        Tensor *t = opt->params[i];
        if (t->grad) {
            for (size_t j = 0; j < t->size; j++) {
                t->grad[j] = 0.0;
            }
        }
    }
}

void optimizer_free(Optimizer *opt) {
    if (!opt) return;
    free(opt->params);
    free(opt);
}

/* --- Adam Optimizer Implementation --- */

AdamOptimizer* optimizer_adam_create(Tensor **params, int count, double lr) {
    AdamOptimizer *opt = (AdamOptimizer*)malloc(sizeof(AdamOptimizer));
    if (!opt) return NULL;

    opt->params = (Tensor**)malloc(sizeof(Tensor*) * count);
    opt->m = (Tensor**)malloc(sizeof(Tensor*) * count);
    opt->v = (Tensor**)malloc(sizeof(Tensor*) * count);

    for (int i = 0; i < count; i++) {
        opt->params[i] = params[i];
        opt->m[i] = tensor_zeros(params[i]->dtype, params[i]->shape, params[i]->ndims, false);
        opt->v[i] = tensor_zeros(params[i]->dtype, params[i]->shape, params[i]->ndims, false);
    }

    opt->count = count;
    opt->lr = lr;
    opt->beta1 = 0.9;
    opt->beta2 = 0.999;
    opt->epsilon = 1e-8;
    opt->t = 0;

    return opt;
}

NcStatus optimizer_adam_step(AdamOptimizer *opt) {
    if (!opt) return NC_ERROR_NULL_POINTER;
    opt->t++;
    
    double bias_correction1 = 1.0 - pow(opt->beta1, opt->t);
    double bias_correction2 = 1.0 - pow(opt->beta2, opt->t);

    for (int i = 0; i < opt->count; i++) {
        Tensor *p = opt->params[i];
        if (!p->grad) continue;

        double *restrict data = (double*)p->data;
        double *restrict grad = (double*)p->grad;
        double *restrict m = (double*)opt->m[i]->data;
        double *restrict v = (double*)opt->v[i]->data;

        for (size_t j = 0; j < p->size; j++) {
            // m = beta1 * m + (1 - beta1) * grad
            m[j] = opt->beta1 * m[j] + (1.0 - opt->beta1) * grad[j];
            // v = beta2 * v + (1 - beta2) * grad^2
            v[j] = opt->beta2 * v[j] + (1.0 - opt->beta2) * grad[j] * grad[j];

            double m_hat = m[j] / bias_correction1;
            double v_hat = v[j] / bias_correction2;

            data[j] -= opt->lr * m_hat / (sqrt(v_hat) + opt->epsilon);
        }
    }
    return NC_SUCCESS;
}

void optimizer_adam_zero_grad(AdamOptimizer *opt) {
    if (!opt) return;
    for (int i = 0; i < opt->count; i++) {
        if (opt->params[i]->grad) {
            memset(opt->params[i]->grad, 0, opt->params[i]->size * sizeof(double));
        }
    }
}

void optimizer_adam_free(AdamOptimizer *opt) {
    if (!opt) return;
    for (int i = 0; i < opt->count; i++) {
        tensor_free(opt->m[i]);
        tensor_free(opt->v[i]);
    }
    free(opt->params);
    free(opt->m);
    free(opt->v);
    free(opt);
}