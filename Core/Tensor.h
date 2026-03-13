#ifndef NEURALC_TENSOR_H
#define NEURALC_TENSOR_H

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define TENSOR_MAX_DIMS 5

typedef enum {
    DTYPE_FLOAT32,
    DTYPE_FLOAT64,
    DTYPE_INT32,
    DTYPE_UINT8
} DataType;

typedef struct Tensor {
    void *data;
    double *grad;
    DataType dtype;
    int shape[TENSOR_MAX_DIMS];
    int strides[TENSOR_MAX_DIMS];
    int ndims;
    size_t size;
    bool requires_grad;

    // Future: struct Node *grad_fn; for Autograd graph tracking
} Tensor;

Tensor* tensor_create(DataType dtype, const int *shape, int ndims, bool requires_grad);

Tensor* tensor_zeros(DataType dtype, const int *shape, int ndims, bool requires_grad);

void tensor_free(Tensor *tensor);
size_t tensor_dtype_size(DataType dtype);

Tensor* tensor_cast(const Tensor *tensor, DataType new_dtype);

Tensor* tensor_add(const Tensor *a, const Tensor *b);
Tensor* tensor_matmul(const Tensor *a, const Tensor *b);

void tensor_print(const Tensor *tensor);

#endif //NEURALC_TENSOR_H