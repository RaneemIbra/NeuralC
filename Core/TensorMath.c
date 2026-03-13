#include "TensorMath.h"
#include <stdio.h>
#include <stdlib.h>

Tensor* tensor_matmul(const Tensor *a, const Tensor *b) {
    if (a->ndims != 2 || b->ndims != 2) return NULL;
    if (a->shape[1] != b->shape[0]) {
        fprintf(stderr, "Error: Incompatible shapes for matmul\n");
        return NULL;
    }

    int res_shape[] = {a->shape[0], b->shape[1]};
    Tensor *res = tensor_zeros(a->dtype, res_shape, 2, a->requires_grad || b->requires_grad);

    int N = a->shape[0];
    int K = a->shape[1];
    int M = b->shape[1];

    if (a->dtype == DTYPE_FLOAT64 && b->dtype == DTYPE_FLOAT64) {
        double *A = (double*)a->data;
        double *B = (double*)b->data;
        double *C = (double*)res->data;

        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                double temp = A[i * K + k];
                for (int j = 0; j < M; j++) {
                    C[i * M + j] += temp * B[k * M + j];
                }
            }
        }
    } else if (a->dtype == DTYPE_FLOAT32 && b->dtype == DTYPE_FLOAT32) {
        float *A = (float*)a->data;
        float *B = (float*)b->data;
        float *C = (float*)res->data;

        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                float temp = A[i * K + k];
                for (int j = 0; j < M; j++) {
                    C[i * M + j] += temp * B[k * M + j];
                }
            }
        }
    }
    // TODO: Add support for other dtype combinations / casting

    return res;
}

Tensor* tensor_add(const Tensor *a, const Tensor *b) {
    if (a->ndims != b->ndims) return NULL;
    for (int i = 0; i < a->ndims; i++) {
        if (a->shape[i] != b->shape[i]) return NULL;
    }

    Tensor *res = tensor_create(a->dtype, a->shape, a->ndims, a->requires_grad || b->requires_grad);

    if (a->dtype == DTYPE_FLOAT64) {
        double *A = (double*)a->data;
        double *B = (double*)b->data;
        double *C = (double*)res->data;
        for (size_t i = 0; i < a->size; i++) C[i] = A[i] + B[i];
    } else if (a->dtype == DTYPE_FLOAT32) {
        float *A = (float*)a->data;
        float *B = (float*)b->data;
        float *C = (float*)res->data;
        for (size_t i = 0; i < a->size; i++) C[i] = A[i] + B[i];
    }

    return res;
}
