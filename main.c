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

void perform_step(Model *model, void *optimizer, bool is_adam, Tensor *x, Tensor *y_true,
                  double (*loss_fn)(const Tensor*, const Tensor*),
                  NcStatus (*loss_derivative)(const Tensor*, const Tensor*, Tensor**)) {
    Tensor *y_pred = NULL;
    model_forward(model, x, &y_pred);

    double loss = loss_fn(y_pred, y_true);

    Tensor *grad_output = NULL;
    loss_derivative(y_pred, y_true, &grad_output);

    NcGraph *backprop_graph = nc_graph_create();
    model_backward(model, grad_output, backprop_graph);

    if (is_adam) {
        optimizer_adam_step((AdamOptimizer*)optimizer);
    } else {
        optimizer_sgd_step((SGDOptimizer*)optimizer);
    }

    nc_graph_free(backprop_graph);
    tensor_free(grad_output);
    tensor_free(y_pred);
}

void perform_epochs(Model *model, void *optimizer, bool is_adam, Tensor *x, Tensor *y_true,
                    double (*loss_fn)(const Tensor*, const Tensor*),
                    NcStatus (*loss_derivative)(const Tensor*, const Tensor*, Tensor**),
                    int epochs) {
    for (int epoch = 1; epoch <= epochs; epoch++) {
        Tensor *y_pred = NULL;
        model_forward(model, x, &y_pred);
        double loss = loss_fn(y_pred, y_true);
        printf("    Epoch %d - Loss: %.6f\n", epoch, loss);

        Tensor *grad_output = NULL;
        loss_derivative(y_pred, y_true, &grad_output);

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
}

void ffn_sgd_relu_xavier() {
    printf("\nFFN: SGD + ReLU + Xavier Uniform + MSE\n");
    Model *model = model_create();
    model_add(model, linear_layer_create(5, 16, relu_activation, "h1", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, linear_layer_create(16, 8, relu_activation, "h2", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, linear_layer_create(8, 2, linear_activation, "out", get_initializer(INIT_XAVIER_UNIFORM)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    SGDOptimizer *opt = optimizer_sgd_create(params, p_count, 0.1);

    int in_s[] = {5, 1};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
    for (int i = 0; i < 5; i++) ((double*)x->data)[i] = 0.2 * (i + 1);

    int t_s[] = {2, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[0] = 1.5; ((double*)y->data)[1] = -0.5;

    perform_epochs(model, opt, false, x, y, mse_loss, mse_loss_derivative, 8);

    tensor_free(x); tensor_free(y);
    optimizer_sgd_free(opt); free(params); model_free(model);
}

void ffn_adam_sigmoid_he() {
    printf("\nFFN: Adam + Sigmoid + He Normal + Binary CE\n");
    Model *model = model_create();
    model_add(model, linear_layer_create(4, 12, sigmoid_activation, "h1", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(12, 6, sigmoid_activation, "h2", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(6, 1, sigmoid_activation, "out", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    AdamOptimizer *opt = optimizer_adam_create(params, p_count, 0.001);

    int in_s[] = {4, 1};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
    ((double*)x->data)[0] = 0.7; ((double*)x->data)[1] = 0.3;
    ((double*)x->data)[2] = 0.9; ((double*)x->data)[3] = 0.1;

    int t_s[] = {1, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[0] = 0.8;

    perform_epochs(model, opt, true, x, y, binary_CE_loss, binary_CE_loss_derivative, 8);

    tensor_free(x); tensor_free(y);
    optimizer_adam_free(opt); free(params); model_free(model);
}

void ffn_sgd_tanh_lecun() {
    printf("\nFFN: SGD + Tanh + LeCun Normal + MAE\n");
    Model *model = model_create();
    model_add(model, linear_layer_create(6, 14, tanh_activation, "h1", get_initializer(INIT_LECUN_NORMAL)));
    model_add(model, linear_layer_create(14, 8, tanh_activation, "h2", get_initializer(INIT_LECUN_NORMAL)));
    model_add(model, linear_layer_create(8, 3, linear_activation, "out", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    SGDOptimizer *opt = optimizer_sgd_create(params, p_count, 0.05);

    int in_s[] = {6, 1};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
    for (int i = 0; i < 6; i++) ((double*)x->data)[i] = 0.1 * i - 0.2;

    int t_s[] = {3, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[0] = 0.5; ((double*)y->data)[1] = -0.3; ((double*)y->data)[2] = 0.7;

    perform_epochs(model, opt, false, x, y, mae_loss, mse_loss_derivative, 8);

    tensor_free(x); tensor_free(y);
    optimizer_sgd_free(opt); free(params); model_free(model);
}

void ffn_adam_swish_he_with_dropout() {
    printf("\nFFN: Adam + Swish + He Uniform + Dropout(0.3) + Huber Loss\n");
    Model *model = model_create();
    model_add(model, linear_layer_create(8, 20, swish_activation, "h1", get_initializer(INIT_HE_UNIFORM)));
    model_add(model, dropout_layer_create(0.3, "drop1"));
    model_add(model, linear_layer_create(20, 12, swish_activation, "h2", get_initializer(INIT_HE_UNIFORM)));
    model_add(model, dropout_layer_create(0.3, "drop2"));
    model_add(model, linear_layer_create(12, 4, linear_activation, "out", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    AdamOptimizer *opt = optimizer_adam_create(params, p_count, 0.001);

    int in_s[] = {8, 1};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
    for (int i = 0; i < 8; i++) ((double*)x->data)[i] = 0.15 * (i - 3.5);

    int t_s[] = {4, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[0] = 1.0; ((double*)y->data)[2] = 0.5;

    perform_epochs(model, opt, true, x, y, huber_loss, mse_loss_derivative, 8);

    tensor_free(x); tensor_free(y);
    optimizer_adam_free(opt); free(params); model_free(model);
}

void ffn_sgd_leaky_relu_mse() {
    printf("\nFFN: SGD + LeakyReLU + Xavier Normal + MSE\n");
    Model *model = model_create();
    model_add(model, linear_layer_create(10, 24, leaky_relu_activation, "h1", get_initializer(INIT_XAVIER_NORMAL)));
    model_add(model, linear_layer_create(24, 16, leaky_relu_activation, "h2", get_initializer(INIT_XAVIER_NORMAL)));
    model_add(model, linear_layer_create(16, 5, linear_activation, "out", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    SGDOptimizer *opt = optimizer_sgd_create(params, p_count, 0.08);

    int in_s[] = {10, 1};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
    for (int i = 0; i < 10; i++) ((double*)x->data)[i] = 0.1 * (i - 4.5);

    int t_s[] = {5, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[2] = 1.0;

    perform_epochs(model, opt, false, x, y, mse_loss, mse_loss_derivative, 8);

    tensor_free(x); tensor_free(y);
    optimizer_sgd_free(opt); free(params); model_free(model);
}

void cnn_conv_relu_adam_maxpool() {
    printf("\nCNN: Conv + ReLU + MaxPool + Adam + Xavier + MSE\n");
    Model *model = model_create();
    model_add(model, conv2d_layer_create(1, 8, 3, 3, 1, 1, 1, 1, "conv1", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, activation_layer_create(relu_activation, relu_derivative, "relu1"));
    model_add(model, pooling_layer_create(POOL_MAX, 2, 2, 2, 2, "pool1"));
    model_add(model, flatten_layer_create("flatten"));
    model_add(model, linear_layer_create(8*8*8, 16, relu_activation, "fc1", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(16, 4, linear_activation, "fc_out", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    AdamOptimizer *opt = optimizer_adam_create(params, p_count, 0.001);

    int in_s[] = {1, 16, 16};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 3, false);
    for (size_t i = 0; i < x->size; i++) ((double*)x->data)[i] = 0.4 + 0.05 * (i % 5);

    int t_s[] = {4, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[1] = 0.9;

    perform_epochs(model, opt, true, x, y, mse_loss, mse_loss_derivative, 8);

    tensor_free(x); tensor_free(y);
    optimizer_adam_free(opt); free(params); model_free(model);
}

void cnn_multi_conv_sigmoid_avgpool_sgd() {
    printf("\nCNN: MultiConv + Sigmoid + AvgPool + SGD + He + Binary CE\n");
    Model *model = model_create();
    model_add(model, conv2d_layer_create(3, 6, 3, 3, 1, 1, 1, 1, "conv1", get_initializer(INIT_HE_NORMAL)));
    model_add(model, activation_layer_create(sigmoid_activation, sigmoid_derivative, "sigmoid1"));
    model_add(model, pooling_layer_create(POOL_AVERAGE, 2, 2, 2, 2, "avgpool1"));
    model_add(model, conv2d_layer_create(6, 12, 3, 3, 1, 1, 1, 1, "conv2", get_initializer(INIT_HE_NORMAL)));
    model_add(model, activation_layer_create(sigmoid_activation, sigmoid_derivative, "sigmoid2"));
    model_add(model, pooling_layer_create(POOL_AVERAGE, 2, 2, 2, 2, "avgpool2"));
    model_add(model, flatten_layer_create("flatten"));
    model_add(model, linear_layer_create(12*4*4, 8, sigmoid_activation, "fc1", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, linear_layer_create(8, 1, sigmoid_activation, "fc_out", get_initializer(INIT_XAVIER_UNIFORM)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    SGDOptimizer *opt = optimizer_sgd_create(params, p_count, 0.01);

    int in_s[] = {3, 16, 16};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 3, false);
    for (size_t i = 0; i < x->size; i++) ((double*)x->data)[i] = 0.5 + 0.1 * ((i % 7) * 0.07);

    int t_s[] = {1, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[0] = 0.7;

    perform_epochs(model, opt, false, x, y, binary_CE_loss, binary_CE_loss_derivative, 8);

    tensor_free(x); tensor_free(y);
    optimizer_sgd_free(opt); free(params); model_free(model);
}

void cnn_deep_with_dropout_tanh() {
    printf("\nCNN: Deep Conv + Tanh + Dropout(0.25) + Adam + LeCun + Huber Loss\n");
    Model *model = model_create();
    model_add(model, conv2d_layer_create(1, 8, 3, 3, 1, 1, 1, 1, "conv1", get_initializer(INIT_LECUN_NORMAL)));
    model_add(model, activation_layer_create(tanh_activation, tanh_derivative, "tanh1"));
    model_add(model, dropout_layer_create(0.25, "drop1"));
    model_add(model, pooling_layer_create(POOL_MAX, 2, 2, 2, 2, "pool1"));
    model_add(model, conv2d_layer_create(8, 16, 3, 3, 1, 1, 1, 1, "conv2", get_initializer(INIT_LECUN_NORMAL)));
    model_add(model, activation_layer_create(tanh_activation, tanh_derivative, "tanh2"));
    model_add(model, dropout_layer_create(0.25, "drop2"));
    model_add(model, global_average_pooling_layer_create("gap"));
    model_add(model, flatten_layer_create("flatten"));
    model_add(model, linear_layer_create(16, 6, linear_activation, "fc_out", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    AdamOptimizer *opt = optimizer_adam_create(params, p_count, 0.0005);

    int in_s[] = {1, 16, 16};
    Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 3, false);
    for (size_t i = 0; i < x->size; i++) ((double*)x->data)[i] = 0.3 + 0.08 * ((i % 11) * 0.09);

    int t_s[] = {6, 1};
    Tensor *y = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
    ((double*)y->data)[2] = 0.8;

    perform_epochs(model, opt, true, x, y, huber_loss, mse_loss_derivative, 8);

    tensor_free(x); tensor_free(y);
    optimizer_adam_free(opt); free(params); model_free(model);
}

int main(void) {
    srand((unsigned int)time(NULL));
    printf("=== NeuralC: Comprehensive Demos ===\n");

    printf("\n--- Feed Forward Networks ---");
    ffn_sgd_relu_xavier();
    ffn_adam_sigmoid_he();
    ffn_sgd_tanh_lecun();
    ffn_adam_swish_he_with_dropout();
    ffn_sgd_leaky_relu_mse();

    printf("\n--- Convolutional Networks ---");
    cnn_conv_relu_adam_maxpool();
    cnn_multi_conv_sigmoid_avgpool_sgd();
    cnn_deep_with_dropout_tanh();

    printf("\n\nAll demonstrations completed successfully!\n");
    return 0;
}
