#include "TensorMath.h"
#include <stdio.h>
#include <stdlib.h>

NcStatus tensor_matmul(const Tensor *a, const Tensor *b, Tensor **result) {
    if (!a || !b || !result) return NC_ERROR_NULL_POINTER;
    if (a->ndims != 2 || b->ndims != 2) return NC_ERROR_SHAPE_MISMATCH;
    if (a->shape[1] != b->shape[0]) {
        NC_REPORT_ERROR(NC_ERROR_SHAPE_MISMATCH);
        return NC_ERROR_SHAPE_MISMATCH;
    }

    int res_shape[] = {a->shape[0], b->shape[1]};
    Tensor *res = tensor_zeros(a->dtype, res_shape, 2, a->requires_grad || b->requires_grad);
    if (!res) return NC_ERROR_OUT_OF_MEMORY;

    const int N = a->shape[0];
    const int K = a->shape[1];
    const int M = b->shape[1];

    if (a->dtype == DTYPE_FLOAT64 && b->dtype == DTYPE_FLOAT64) {
        const double *restrict A = (const double*)a->data;
        const double *restrict B = (const double*)b->data;
        double *restrict C = (double*)res->data;

        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                const double temp = A[i * K + k];
                for (int j = 0; j < M; j++) {
                    C[i * M + j] += temp * B[k * M + j];
                }
            }
        }
    } else if (a->dtype == DTYPE_FLOAT32 && b->dtype == DTYPE_FLOAT32) {
        const float *restrict A = (const float*)a->data;
        const float *restrict B = (const float*)b->data;
        float *restrict C = (float*)res->data;

        for (int i = 0; i < N; i++) {
            for (int k = 0; k < K; k++) {
                const float temp = A[i * K + k];
                for (int j = 0; j < M; j++) {
                    C[i * M + j] += temp * B[k * M + j];
                }
            }
        }
    } else {
        tensor_free(res);
        return NC_ERROR_INVALID_DTYPE;
    }

    *result = res;
    return NC_SUCCESS;
}

NcStatus tensor_add(const Tensor *a, const Tensor *b, Tensor **result) {
    if (!a || !b || !result) return NC_ERROR_NULL_POINTER;
    if (a->ndims != b->ndims) return NC_ERROR_SHAPE_MISMATCH;
    for (int i = 0; i < a->ndims; i++) {
        if (a->shape[i] != b->shape[i]) return NC_ERROR_SHAPE_MISMATCH;
    }

    Tensor *res = tensor_create(a->dtype, a->shape, a->ndims, a->requires_grad || b->requires_grad);
    if (!res) return NC_ERROR_OUT_OF_MEMORY;

    if (a->dtype == DTYPE_FLOAT64) {
        const double *restrict A = (const double*)a->data;
        const double *restrict B = (const double*)b->data;
        double *restrict C = (double*)res->data;
        for (size_t i = 0; i < a->size; i++) C[i] = A[i] + B[i];
    } else if (a->dtype == DTYPE_FLOAT32) {
        const float *restrict A = (const float*)a->data;
        const float *restrict B = (const float*)b->data;
        float *restrict C = (float*)res->data;
        for (size_t i = 0; i < a->size; i++) C[i] = A[i] + B[i];
    } else {
        tensor_free(res);
        return NC_ERROR_INVALID_DTYPE;
    }

    *result = res;
    return NC_SUCCESS;
}
