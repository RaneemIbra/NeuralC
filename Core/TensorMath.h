#ifndef NEURALC_TENSORMATH_H
#define NEURALC_TENSORMATH_H

#include "Tensor.h"
#include "Common.h"

/**
 * Standard Operations
 */
NcStatus tensor_matmul(const Tensor *a, const Tensor *b, Tensor **result);
NcStatus tensor_add(const Tensor *a, const Tensor *b, Tensor **result);

/**
 * Specialized Gradients Operations
 */

// result = (trans_a ? a^T : a) @ (trans_b ? b^T : b)
NcStatus tensor_matmul_extended(const Tensor *a, bool trans_a, const Tensor *b, bool trans_b, Tensor **out);

/**
 * im2col: Image to Column optimization for Convolution
 * Transforms a 4D tensor (N, C, H, W) into a 2D matrix for GEMM convolution.
 */
NcStatus nc_im2col(const Tensor *input, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, Tensor **out);

NcStatus nc_col2im(const Tensor *cols, int channels, int height, int width,
                   int kernel_h, int kernel_w, int stride_h, int stride_w,
                   int pad_h, int pad_w, Tensor **out);

// Element-wise addition into dest: dest += src
NcStatus tensor_add_inplace(Tensor *dest, const Tensor *src);

// Element-wise multiplication into dest: dest *= src
NcStatus tensor_mul_inplace(Tensor *dest, const Tensor *src);

// Scale a tensor by a scalar: tensor *= scale
NcStatus tensor_scale(Tensor *t, double scale);

#endif //NEURALC_TENSORMATH_H
