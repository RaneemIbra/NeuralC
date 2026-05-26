#include <stdio.h>
#include "FFN_demos.h"
#include "../Entities/Model/Model.h"
#include "../Entities/Layers/Linear/LinearLayer.h"
#include "../Entities/Layers/Linear/Flatten.h"
#include "../Entities/Layers/Linear/Dropout.h"
#include "../Entities/Layers/Activations/ActivationLayer.h"
#include "../Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "../Utils/Initializers/Initializers.h"
#include "../Utils/LossFunctions/LossFunctions.h"
#include "../Utils/Optimizers/SGD/SGD.h"
#include "../Utils/Optimizers/Adam/Adam.h"
#include "../Core/Tensor.h"
#include "../Core/Graph.h"
#include "../Core/Training.h"

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

void  ffn_sgd_leaky_relu_mse() {
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