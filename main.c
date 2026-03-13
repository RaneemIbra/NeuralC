#include <stdio.h>
#include "Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "Entities/Layers/FeedForward/FeedForward.h"
#include "Utils/Initializers/Initializers.h"
#include "Utils/LossFunctions/LossFunctions.h"
#include "Core/Tensor.h"

int main(void) {
    FeedForward layer;
    const int inputs_dim = 3;
    const int neurons_dim = 2;
    const char* activation_name = "relu";

    init_feed_forward(&layer, inputs_dim, neurons_dim, relu_activation, activation_name, get_initializer(INIT_XAVIER_UNIFORM));

    int in_shape[] = {inputs_dim, 1};
    Tensor *input_tensor = tensor_create(DTYPE_FLOAT64, in_shape, 2, false);
    double *in_data = (double*)input_tensor->data;
    in_data[0] = 1.0; in_data[1] = 2.0; in_data[2] = -1.0;

    int target_shape[] = {neurons_dim, 1};
    Tensor *target_tensor = tensor_create(DTYPE_FLOAT64, target_shape, 2, false);
    double *target_data = (double*)target_tensor->data;
    target_data[0] = 1.0; target_data[1] = 0.0;

    Tensor *output = forward_pass(&layer, input_tensor);

    printf("--- FeedForward Layer (Tensor Edition) ---\n");
    printf("Weights:\n");
    tensor_print(layer.weights);
    
    printf("\nBiases:\n");
    tensor_print(layer.biases);

    printf("\nInput:\n");
    tensor_print(input_tensor);

    printf("\nOutput:\n");
    tensor_print(output);

    double loss = mse_loss(output, target_tensor);
    printf("\nMSE Loss against target [1.0, 0.0]: %f\n", loss);

    tensor_free(input_tensor);
    tensor_free(target_tensor);
    free_feed_forward(&layer);

    printf("\nPipeline test completed successfully!\n");
    return 0;
}
