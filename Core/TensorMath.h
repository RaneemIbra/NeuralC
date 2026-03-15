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
NcStatus tensor_matmul_extended(const Tensor *a, bool trans_a, const Tensor *b, bool trans_b, Tensor **result);

// Element-wise addition into dest: dest += src
NcStatus tensor_add_inplace(Tensor *dest, const Tensor *src);

// Element-wise multiplication into dest: dest *= src
NcStatus tensor_mul_inplace(Tensor *dest, const Tensor *src);

// Scale a tensor by a scalar: tensor *= scale
NcStatus tensor_scale(Tensor *t, double scale);

#endif //NEURALC_TENSORMATH_H
