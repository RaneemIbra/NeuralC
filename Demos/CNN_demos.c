#include <stdio.h>
#include <stdlib.h>
#include "CNN_demos.h"
#include "../Entities/Model/Model.h"
#include "../Entities/Layers/Linear/LinearLayer.h"
#include "../Entities/Layers/Linear/Flatten.h"
#include "../Entities/Layers/Linear/Dropout.h"
#include "../Entities/Layers/Convolution/Conv2d.h"
#include "../Entities/Layers/Pooling/Pooling.h"
#include "../Entities/Layers/Activations/ActivationLayer.h"
#include "../Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "../Utils/Initializers/Initializers.h"
#include "../Utils/LossFunctions/LossFunctions.h"
#include "../Utils/Optimizers/SGD/SGD.h"
#include "../Utils/Optimizers/Adam/Adam.h"
#include "../Core/Tensor.h"
#include "../Core/Graph.h"
#include "../Core/Training.h"

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
