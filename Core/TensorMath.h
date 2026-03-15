#ifndef NEURALC_TENSORMATH_H
#define NEURALC_TENSORMATH_H

#include "Tensor.h"
#include "Common.h"

/**
 * Professional Tensor Math API
 * Returns NcStatus and uses restrict for performance.
 */

NcStatus tensor_matmul(const Tensor *a, const Tensor *b, Tensor **result);

NcStatus tensor_add(const Tensor *a, const Tensor *b, Tensor **result);

#endif //NEURALC_TENSORMATH_H
