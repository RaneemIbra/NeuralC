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

    if (a_cols != b_rows) {
        printf("[TensorMath Error] MatMul Mismatch: A[%d,%d] x B[%d,%d]\n", a_rows, a_cols, b_rows, b_cols);
        return NC_ERROR_SHAPE_MISMATCH;
    }

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

NcStatus nc_im2col(const Tensor *input, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, Tensor **out) {
    if (!input || !out) return NC_ERROR_NULL_POINTER;
    
    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    
    // Result shape: [channels * kernel_h * kernel_w, out_h * out_w]
    int res_shape[] = {channels * kernel_h * kernel_w, out_h * out_w};
    Tensor *res = tensor_zeros(DTYPE_FLOAT64, res_shape, 2, false);
    if (!res) return NC_ERROR_OUT_OF_MEMORY;

    const double *data = (const double*)input->data;
    double *cols = (double*)res->data;

    for (int c = 0; c < channels; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int row = (c * kernel_h + kh) * kernel_w + kw;
                for (int h = 0; h < out_h; h++) {
                    for (int w = 0; w < out_w; w++) {
                        int im_h = h * stride_h + kh - pad_h;
                        int im_w = w * stride_w + kw - pad_w;
                        if (im_h >= 0 && im_h < height && im_w >= 0 && im_w < width) {
                            cols[row * (out_h * out_w) + (h * out_w + w)] = 
                                data[(c * height + im_h) * width + im_w];
                        } else {
                            cols[row * (out_h * out_w) + (h * out_w + w)] = 0;
                        }
                    }
                }
            }
        }
    }

    *out = res;
    return NC_SUCCESS;
}

NcStatus nc_col2im(const Tensor *cols, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, Tensor **out) {
    if (!cols || !out) return NC_ERROR_NULL_POINTER;

    int out_h = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    int out_w = (width + 2 * pad_w - kernel_w) / stride_w + 1;

    int im_shape[] = {channels, height, width};
    Tensor *im = tensor_zeros(DTYPE_FLOAT64, im_shape, 3, false);
    if (!im) return NC_ERROR_OUT_OF_MEMORY;

    double *data = (double*)im->data;
    const double *col_data = (const double*)cols->data;

    for (int c = 0; c < channels; c++) {
        for (int kh = 0; kh < kernel_h; kh++) {
            for (int kw = 0; kw < kernel_w; kw++) {
                int row = (c * kernel_h + kh) * kernel_w + kw;
                for (int h = 0; h < out_h; h++) {
                    for (int w = 0; w < out_w; w++) {
                        int im_h = h * stride_h + kh - pad_h;
                        int im_w = w * stride_w + kw - pad_w;
                        if (im_h >= 0 && im_h < height && im_w >= 0 && im_w < width) {
                            data[(c * height + im_h) * width + im_w] += 
                                col_data[row * (out_h * out_w) + (h * out_w + w)];
                        }
                    }
                }
            }
        }
    }

    *out = im;
    return NC_SUCCESS;
}
