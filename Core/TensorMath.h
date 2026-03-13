#ifndef NEURALC_TENSORMATH_H
#define NEURALC_TENSORMATH_H

#include "Tensor.h"

Tensor* tensor_matmul(const Tensor *a, const Tensor *b);

Tensor* tensor_add(const Tensor *a, const Tensor *b);

#endif //NEURALC_TENSORMATH_H
