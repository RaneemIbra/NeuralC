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
#include "Utils/Visualizer/Visualizer.h"
#include "raylib.h"

int main(void) {
srand(time(NULL));
    
    // 1. Setup Pure FeedForward Model (MLP)
    NcModel *model = model_create();
    model_add(model, linear_layer_create(2, 16, relu_activation, "input_dense", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(16, 8, relu_activation, "hidden_dense", get_initializer(INIT_HE_NORMAL)));
    model_add(model, linear_layer_create(8, 1, linear_activation, "output_dense", get_initializer(INIT_XAVIER_NORMAL)));

    int p_count;
    Tensor **params = model_get_parameters(model, &p_count);
    AdamOptimizer *opt = optimizer_adam_create(params, p_count, 0.01);

    InitWindow(1280, 720, "NeuralC - Real-time Visualization");
    SetTargetFPS(60);

    VisState state = {
        .loss_history = calloc(1000, sizeof(float)),
        .history_count = 0,
        .history_capacity = 1000,
        .is_training = true,
        .camera = { 
            .offset = (Vector2){ 400, 360 }, 
            .target = (Vector2){ 0, 110 }, 
            .rotation = 0.0f, 
            .zoom = 1.0f 
        },
        .selected_layer = -1,
        .hovered_layer = -1
    };

    while (!WindowShouldClose()) {
        int in_s[] = {2, 1}; // Input size 2
        Tensor *x = tensor_zeros(DTYPE_FLOAT64, in_s, 2, false);
        ((double*)x->data)[0] = (double)rand()/RAND_MAX;
        
        int t_s[] = {1, 1};
        Tensor *y_true = tensor_zeros(DTYPE_FLOAT64, t_s, 2, false);
        ((double*)y_true->data)[0] = 0.5; // Target

        Tensor *y_pred = NULL;
        model_forward(model, x, &y_pred);
        double loss = mse_loss(y_pred, y_true);
        
        visualizer_add_loss(&state, (float)loss);

        Tensor *grad_output = NULL;
        mse_loss_derivative(y_pred, y_true, &grad_output);
        NcGraph *g = nc_graph_create();
        model_backward(model, grad_output, g);
        optimizer_adam_step(opt);

        visualizer_run(model, &state);

        nc_graph_free(g);
        tensor_free(grad_output);
        tensor_free(x);
        tensor_free(y_true);
    }

    CloseWindow();
    return 0;
}
