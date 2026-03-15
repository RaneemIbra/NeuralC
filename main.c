#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Entities/Model/Model.h"
#include "Entities/Layers/Linear/LinearLayer.h"
#include "Entities/Layers/Linear/Flatten.h"
#include "Entities/Layers/Linear/Dropout.h"
#include "Entities/Layers/Convolution/Conv2d.h"
#include "Entities/Layers/Pooling/Pooling.h"
#include "Entities/Layers/Activations/ActivationLayer.h"
#include "Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "Utils/Initializers/Initializers.h"
#include "Utils/LossFunctions/LossFunctions.h"
#include "Utils/Optimizers/SGD/SGD.h"
#include "Utils/Optimizers/Adam/Adam.h"
#include "Core/Tensor.h"
#include "Core/Common.h"
#include "Core/Graph.h"

// Helper to run a training step
void perform_step(Model *model, void *optimizer, bool is_adam, Tensor *x, Tensor *y_true) {
    Tensor *y_pred = NULL;
    model_forward(model, x, &y_pred);
    
    double loss = mse_loss(y_pred, y_true);
    printf("    [Step] Loss: %.6f\n", loss);

    Tensor *grad_output = NULL;
    mse_loss_derivative(y_pred, y_true, &grad_output);
    
    NcGraph *backprop_graph = nc_graph_create();
    model_backward(model, grad_output, backprop_graph);

    if (is_adam) {
        optimizer_adam_step((AdamOptimizer*)optimizer);
    } else {
        optimizer_sgd_step((SGDOptimizer*)optimizer);
    }

    nc_graph_free(backprop_graph);
    tensor_free(grad_output);
}

void run_ffn_sgd_demo() {
    printf("\n--- Demo 1: Simple FFN with SGD ---\n");
    Model *model = model_create();
    model_add(model, linear_layer_create(2, 4, relu_activation, "hidden1", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, linear_layer_create(4, 1, linear_activation, "output", get_initializer(INIT_XAVIER_UNIFORM)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    SGDOptimizer *opt = optimizer_sgd_create(params, p_count, 0.1);

    int in_s[] = {2, 1};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
    ((double*)x->data)[0] = 1.0; ((double*)x->data)[1] = 0.5;

    int t_s[] = {1, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[0] = 5.0;

    perform_step(model, opt, false, x, y);

    tensor_free(x); tensor_free(y);
    optimizer_sgd_free(opt); free(params); model_free(model);
    printf("Demo 1: Success.\n");
}

void run_ffn_adam_demo() {
    printf("\n--- Demo 2: Simple FFN with Adam ---\n");
    Model *model = model_create();
    model_add(model, linear_layer_create(2, 4, relu_activation, "hidden1", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(4, 1, linear_activation, "output", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    AdamOptimizer *opt = optimizer_adam_create(params, p_count, 0.001);

    int in_s[] = {2, 1};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
    int t_s[] = {1, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);

    perform_step(model, opt, true, x, y);

    tensor_free(x); tensor_free(y);
    optimizer_adam_free(opt); free(params); model_free(model);
    printf("Demo 2: Success.\n");
}

void run_deep_ffn_demo() {
    printf("\n--- Demo 3: Deep FFN (5 Layers) ---\n");
    Model *model = model_create();
    model_add(model, linear_layer_create(10, 64, relu_activation, "h1", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(64, 64, relu_activation, "h2", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(64, 32, relu_activation, "h3", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(32, 16, relu_activation, "h4", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(16, 2, linear_activation, "out", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    AdamOptimizer *opt = optimizer_adam_create(params, p_count, 0.001);

    int in_s[] = {10, 1};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
    int t_s[] = {2, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);

    perform_step(model, opt, true, x, y);

    tensor_free(x); tensor_free(y);
    optimizer_adam_free(opt); free(params); model_free(model);
    printf("Demo 3: Success.\n");
}

void run_basic_cnn_demo() {
    printf("\n--- Demo 4: Basic CNN (Conv -> Pool -> FC) ---\n");
    Model *model = model_create();
    // Input [1, 14, 14]
    model_add(model, conv2d_layer_create(1, 4, 3, 3, 1, 1, 1, 1, "conv1", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, activation_layer_create(relu_activation, relu_derivative, "relu1"));
    model_add(model, pooling_layer_create(POOL_MAX, 2, 2, 2, 2, "pool1")); // Output [4, 7, 7]
    model_add(model, flatten_layer_create("flatten"));
    model_add(model, linear_layer_create(4*7*7, 10, linear_activation, "fc1", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    SGDOptimizer *opt = optimizer_sgd_create(params, p_count, 0.01);

    int in_s[] = {1, 14, 14};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 3, false);
    int t_s[] = {10, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);

    perform_step(model, opt, false, x, y);

    tensor_free(x); tensor_free(y);
    optimizer_sgd_free(opt); free(params); model_free(model);
    printf("Demo 4: Success.\n");
}

void run_advanced_cnn_demo() {
    printf("\n--- Demo 5: Advanced CNN (Dropout + GAP) ---\n");
    Model *model = model_create();
    // Input [3, 32, 32]
    model_add(model, conv2d_layer_create(3, 8, 3, 3, 1, 1, 1, 1, "conv1", get_initializer(INIT_HE_NORMAL)));
    model_add(model, activation_layer_create(relu_activation, relu_derivative, "relu1"));
    model_add(model, dropout_layer_create(0.2, "drop1"));
    model_add(model, conv2d_layer_create(8, 16, 3, 3, 1, 1, 1, 1, "conv2", get_initializer(INIT_HE_NORMAL)));
    model_add(model, activation_layer_create(relu_activation, relu_derivative, "relu2"));
    model_add(model, global_average_pooling_layer_create("gap")); // Output [16, 1, 1]
    model_add(model, flatten_layer_create("flatten"));
    model_add(model, linear_layer_create(16, 5, linear_activation, "fc_out", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    AdamOptimizer *opt = optimizer_adam_create(params, p_count, 0.001);

    int in_s[] = {3, 32, 32};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 3, false);
    int t_s[] = {5, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);

    perform_step(model, opt, true, x, y);

    tensor_free(x); tensor_free(y);
    optimizer_adam_free(opt); free(params); model_free(model);
    printf("Demo 5: Success.\n");
}

int main(void) {
    srand((unsigned int)time(NULL));
    printf("=== NeuralC: Comprehensive Model Suite ===\n");

    run_ffn_sgd_demo();
    run_ffn_adam_demo();
    run_deep_ffn_demo();
    run_basic_cnn_demo();
    run_advanced_cnn_demo();

    printf("\nAll demonstrations completed successfully!\n");
    return 0;
}
