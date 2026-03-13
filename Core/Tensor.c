#include "Tensor.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

size_t tensor_dtype_size(DataType dtype) {
    switch (dtype) {
        case DTYPE_FLOAT32: return sizeof(float);
        case DTYPE_FLOAT64: return sizeof(double);
        case DTYPE_INT32:   return sizeof(int32_t);
        case DTYPE_UINT8:   return sizeof(uint8_t);
        default:            return 0;
    }
}

Tensor* tensor_create(DataType dtype, const int *shape, int ndims, bool requires_grad) {
    if (ndims > TENSOR_MAX_DIMS) return NULL;

    Tensor *t = (Tensor*)malloc(sizeof(Tensor));
    if (!t) return NULL;

    t->dtype = dtype;
    t->ndims = ndims;
    t->requires_grad = requires_grad;
    t->size = 1;

    for (int i = 0; i < ndims; i++) {
        t->shape[i] = shape[i];
        t->size *= shape[i];
    }

    int stride = 1;
    for (int i = ndims - 1; i >= 0; i--) {
        t->strides[i] = stride;
        stride *= t->shape[i];
    }

    size_t bytes = t->size * tensor_dtype_size(dtype);
    t->data = malloc(bytes);
    if (!t->data) {
        free(t);
        return NULL;
    }

    if (requires_grad) {
        t->grad = (double*)calloc(t->size, sizeof(double));
        if (!t->grad) {
            free(t->data);
            free(t);
            return NULL;
        }
    } else {
        t->grad = NULL;
    }

    return t;
}

Tensor* tensor_zeros(DataType dtype, const int *shape, int ndims, bool requires_grad) {
    Tensor *t = tensor_create(dtype, shape, ndims, requires_grad);
    if (t) {
        memset(t->data, 0, t->size * tensor_dtype_size(dtype));
    }
    return t;
}

void tensor_free(Tensor *tensor) {
    if (!tensor) return;
    if (tensor->data) free(tensor->data);
    if (tensor->grad) free(tensor->grad);
    free(tensor);
}

void tensor_print(const Tensor *tensor) {
    if (!tensor) return;
    printf("Tensor(shape=[");
    for (int i = 0; i < tensor->ndims; i++) {
        printf("%d%s", tensor->shape[i], (i == tensor->ndims - 1) ? "" : ", ");
    }
    printf("], dtype=%d, size=%zu)\n", tensor->dtype, tensor->size);

    if (tensor->ndims <= 2) {
        int rows = (tensor->ndims == 2) ? tensor->shape[0] : 1;
        int cols = (tensor->ndims == 2) ? tensor->shape[1] : tensor->shape[0];

        for (int i = 0; i < rows; i++) {
            printf("  [");
            for (int j = 0; j < cols; j++) {
                size_t idx = i * (tensor->ndims == 2 ? tensor->strides[0] : 0) + j;
                if (tensor->dtype == DTYPE_FLOAT64) 
                    printf("%.4f ", ((double*)tensor->data)[idx]);
                else if (tensor->dtype == DTYPE_FLOAT32)
                    printf("%.4f ", ((float*)tensor->data)[idx]);
                else if (tensor->dtype == DTYPE_INT32)
                    printf("%d ", ((int32_t*)tensor->data)[idx]);
                else if (tensor->dtype == DTYPE_UINT8)
                    printf("%u ", ((uint8_t*)tensor->data)[idx]);
            }
            printf("]\n");
        }
    }
}
