#ifndef NEURALC_TENSORMATH_H
#define NEURALC_TENSORMATH_H

#include "Tensor.h"
#include "Common.h"

NcStatus tensor_matmul(const Tensor *a, const Tensor *b, Tensor **result);
NcStatus tensor_add(const Tensor *a, const Tensor *b, Tensor **result);

NcStatus tensor_matmul_extended(const Tensor *a, bool trans_a, const Tensor *b, bool trans_b, Tensor **out);

NcStatus nc_im2col(const Tensor *input, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, Tensor **out);

NcStatus nc_col2im(const Tensor *cols, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, Tensor **out);

NcStatus tensor_add_inplace(Tensor *dest, const Tensor *src);

NcStatus tensor_mul_inplace(Tensor *dest, const Tensor *src);

NcStatus tensor_scale(Tensor *t, double scale);

#endif //NEURALC_TENSORMATH_H
