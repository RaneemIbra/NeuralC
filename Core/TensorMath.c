#include "TensorMath.h"
#include <stdio.h>
#include <stdlib.h>

NcStatus tensor_matmul(const Tensor *a, const Tensor *b, Tensor **result) {
    return tensor_matmul_extended(a, false, b, false, result);
}

NcStatus tensor_matmul_extended(const Tensor *a, bool trans_a, const Tensor *b, bool trans_b, Tensor **result) {
    if (!a || !b || !result) return NC_ERROR_NULL_POINTER;
    
    int a_rows = trans_a ? a->shape[1] : a->shape[0];
    int a_cols = trans_a ? a->shape[0] : a->shape[1];
    int b_rows = trans_b ? b->shape[1] : b->shape[0];
    int b_cols = trans_b ? b->shape[0] : b->shape[1];

    if (a_cols != b_rows) return NC_ERROR_SHAPE_MISMATCH;

    int res_shape[] = {a_rows, b_cols};
    Tensor *res = tensor_zeros(a->dtype, res_shape, 2, a->requires_grad || b->requires_grad);
    if (!res) return NC_ERROR_OUT_OF_MEMORY;

    if (a->dtype == DTYPE_FLOAT64 && b->dtype == DTYPE_FLOAT64) {
        const double *restrict A = (const double*)a->data;
        const double *restrict B = (const double*)b->data;
        double *restrict C = (double*)res->data;

        // Basic implementation with trans option (could be further optimized with tiling)
        for (int i = 0; i < a_rows; i++) {
            for (int k = 0; k < a_cols; k++) {
                double val_a = trans_a ? A[k * a_rows + i] : A[i * a_cols + k];
                for (int j = 0; j < b_cols; j++) {
                    double val_b = trans_b ? B[j * b_rows + k] : B[k * b_cols + j];
                    C[i * b_cols + j] += val_a * val_b;
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
    if (a->size != b->size) return NC_ERROR_SHAPE_MISMATCH;

    Tensor *res = tensor_create(a->dtype, a->shape, a->ndims, a->requires_grad || b->requires_grad);
    if (!res) return NC_ERROR_OUT_OF_MEMORY;

    double *A = (double*)a->data;
    double *B = (double*)b->data;
    double *C = (double*)res->data;
    for (size_t i = 0; i < a->size; i++) C[i] = A[i] + B[i];

    *result = res;
    return NC_SUCCESS;
}

NcStatus tensor_add_inplace(Tensor *dest, const Tensor *src) {
    if (!dest || !src) return NC_ERROR_NULL_POINTER;
    if (dest->size != src->size) return NC_ERROR_SHAPE_MISMATCH;

    double *D = (double*)dest->data;
    double *S = (double*)src->data;
    for (size_t i = 0; i < dest->size; i++) D[i] += S[i];

    return NC_SUCCESS;
}

NcStatus tensor_mul_inplace(Tensor *dest, const Tensor *src) {
    if (!dest || !src) return NC_ERROR_NULL_POINTER;
    if (dest->size != src->size) return NC_ERROR_SHAPE_MISMATCH;

    double *D = (double*)dest->data;
    double *S = (double*)src->data;
    for (size_t i = 0; i < dest->size; i++) D[i] *= S[i];

    return NC_SUCCESS;
}

NcStatus tensor_scale(Tensor *t, double scale) {
    if (!t) return NC_ERROR_NULL_POINTER;
    double *D = (double*)t->data;
    for (size_t i = 0; i < t->size; i++) D[i] *= scale;
    return NC_SUCCESS;
}
