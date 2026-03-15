#include <stdio.h>
#include <stdlib.h>
#include "Entities/Model/Model.h"
#include "Entities/Layers/Linear/LinearLayer.h"
#include "Entities/Layers/Linear/Flatten.h"
#include "Entities/Layers/Convolution/Conv2d.h"
#include "Entities/Layers/Pooling/Pooling.h"
#include "Entities/Layers/Activations/ActivationLayer.h"
#include "Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "Utils/Initializers/Initializers.h"
#include "Utils/LossFunctions/LossFunctions.h"
#include "Utils/Optimizers/SGD/SGD.h"
#include "Core/Tensor.h"
#include "Core/Common.h"
#include "Core/Graph.h"

int main(void) {
    printf("=== NeuralC: CNN Demonstration ===\n");

    // 1. Create a CNN Model
    // Input: [1, 28, 28] (MNIST-like)
    Model *model = model_create();
    
    // Conv: In=1, Out=4, Kernel=3x3, Stride=1, Pad=1 -> Out Shape: [4, 28, 28]
    model_add(model, conv2d_layer_create(1, 4, 3, 3, 1, 1, 1, 1, "conv1", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, activation_layer_create(relu_activation, relu_derivative, "relu1"));
    
    // Pool: Max 2x2, Stride 2 -> Out Shape: [4, 14, 14]
    model_add(model, pooling_layer_create(POOL_MAX, 2, 2, 2, 2, "pool1"));
    
    // Flatten: 4 * 14 * 14 = 784 -> Out Shape: [784, 1]
    model_add(model, flatten_layer_create("flatten"));
    
    // Linear: 784 -> 10 (Digits)
    model_add(model, linear_layer_create(784, 10, linear_activation, "fc1", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, activation_layer_create(linear_activation, linear_derivative, "linear_out"));

    printf("CNN Model structured successfully.\n");

    // 2. Setup Optimizer
    int param_count = 0;
    Tensor **params = model_get_parameters(model, &param_count);
    SGDOptimizer *optimizer = optimizer_sgd_create(params, param_count, 0.01);
    printf("Parameters to train: %d\n", param_count);

    // 3. Create Dummy Data (Input [1, 28, 28], Target [10, 1])
    int in_shape[] = {1, 28, 28};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_shape, 3, false);
    
    int target_shape[] = {10, 1};
    Tensor *y_true = tensor_zeros(DTYPE_FLOAT64, target_shape, 2, false);
    ((double*)y_true->data)[3] = 1.0; // Assume target is digit '3'

    printf("\nPerforming single Training Step...\n");

    // A. Forward Pass
    Tensor *y_pred = NULL;
    NcStatus status = model_forward(model, x, &y_pred);
    if (status != NC_SUCCESS) {
        printf("Forward pass failed with status %d\n", status);
        return 1;
    }
    printf("Forward Pass: SUCCESS. Output size: %zu\n", y_pred->size);

    // B. Calculate Loss
    double loss = mse_loss(y_pred, y_true);
    printf("Initial Loss: %.6f\n", loss);

    // C. Backward Pass
    Tensor *grad_output = NULL;
    mse_loss_derivative(y_pred, y_true, &grad_output);
    
    NcGraph *backprop_graph = nc_graph_create();
    status = model_backward(model, grad_output, backprop_graph);
    if (status != NC_SUCCESS) {
        printf("Backward pass failed with status %d\n", status);
    } else {
        printf("Backward Pass: SUCCESS.\n");
    }

    // D. Step
    optimizer_sgd_step(optimizer);
    printf("Optimizer step completed.\n");

    // 4. Cleanup
    tensor_free(x);
    tensor_free(y_true);
    optimizer_sgd_free(optimizer);
    free(params);
    nc_graph_free(backprop_graph);
    tensor_free(grad_output);
    model_free(model);

    printf("\nCNN verification completed successfully!\n");
    return 0;
}
