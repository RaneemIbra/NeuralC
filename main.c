#include <stdio.h>
#include <stdlib.h>
#include "Entities/Model/Model.h"
#include "Entities/Layers/FeedForward/FeedForward.h"
#include "Entities/Layers/ActivationLayer.h"
#include "Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "Utils/Initializers/Initializers.h"
#include "Utils/LossFunctions/LossFunctions.h"
#include "Utils/Optimizers/Optimizers.h"
#include "Core/Tensor.h"
#include "Core/Common.h"
#include "Core/Graph.h"

int main(void) {
    printf("=== NeuralC: Training Demonstration ===\n");

    // 1. Create a simple Model: y = Wx + b
    Model *model = model_create();
    // Linear regression: Dense(1, 1) + Linear activation
    model_add(model, dense_layer_create(1, 1, linear_activation, "dense1", get_initializer(INIT_XAVIER_UNIFORM)));
    model_add(model, activation_layer_create(linear_activation, linear_derivative, "linear"));

    // 2. Setup Optimizer
    int param_count = 0;
    Tensor **params = model_get_parameters(model, &param_count);
    Optimizer *optimizer = optimizer_sgd_create(params, param_count, 0.05);

    printf("Model ready. Parameters to train: %d\n", param_count);

    // 3. Create Training Data: y = 2x + 1
    int in_shape[] = {1, 1};
    Tensor *x = tensor_create(DTYPE_FLOAT64, in_shape, 2, false);
    Tensor *y_true = tensor_create(DTYPE_FLOAT64, in_shape, 2, false);

    printf("\nStarting Training (100 epochs)...\n");

    for (int epoch = 0; epoch < 100; epoch++) {
        double total_loss = 0;
        
        // Simulating a small dataset: { (1,3), (2,5), (3,7), (4,9) }
        double inputs[] = {1.0, 2.0, 3.0, 4.0};
        double targets[] = {3.0, 5.0, 7.0, 9.0};

        for (int i = 0; i < 4; i++) {
            ((double*)x->data)[0] = inputs[i];
            ((double*)y_true->data)[0] = targets[i];

            // A. Forward Pass
            Tensor *y_pred = NULL;
            model_forward(model, x, &y_pred);

            // B. Calculate Loss
            double loss = mse_loss(y_pred, y_true);
            total_loss += loss;

            // C. Backward Pass
            Tensor *grad_output = NULL;
            mse_loss_derivative(y_pred, y_true, &grad_output);
            
            NcGraph *backprop_graph = nc_graph_create();
            model_backward(model, grad_output, backprop_graph);

            // D. Optimizer Step
            optimizer_sgd_step(optimizer);

            // E. Cleanup
            optimizer_zero_grad(optimizer);
            nc_graph_free(backprop_graph);
            tensor_free(grad_output);
        }

        if (epoch % 20 == 0) {
            printf("Epoch %d: Loss = %.6f\n", epoch, total_loss / 4.0);
        }
    }

    // 4. Test the model
    printf("\nTesting model (Input: 5.0, Expected: 11.0)...\n");
    ((double*)x->data)[0] = 5.0;
    Tensor *final_out = NULL;
    model_forward(model, x, &final_out);
    printf("Output: %.4f\n", ((double*)final_out->data)[0]);

    // 5. Cleanup
    tensor_free(x);
    tensor_free(y_true);
    optimizer_free(optimizer);
    free(params);
    model_free(model);

    printf("\nTraining test completed successfully!\n");
    return 0;
}
