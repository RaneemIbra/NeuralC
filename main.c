#include <stdio.h>
#include "Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "Entities/Layers/FeedForward/FeedForward.h"
#include "Utils/Initializers/Initializers.h"
#include "Utils/LossFunctions/LossFunctions.h"

int main(void) {
    FeedForward layer, layer2;
    const int inputs = 3;
    const int neurons = 2;
    const char* activation_name = "relu";

    init_feed_forward(&layer, inputs, neurons, relu_activation, activation_name ,get_initializer(INIT_XAVIER_UNIFORM));
    init_feed_forward(&layer2, inputs, neurons, relu_activation, activation_name ,get_initializer(INIT_HE_UNIFORM));

    const double in[] = {1.0, 2.0, -1.0};
    const double target[] = {1.0, 0.0};
    forward_pass(&layer, in);
    forward_pass(&layer2, in);

    printf("Weights initialized using Xavier:\n");
    for (int j = 0; j < layer.neurons_amount; ++j) {
        for (int i = 0; i < layer.inputs_amount; ++i) {
            printf("W[%d][%d] = %f\n", j, i, layer.weights[j * inputs + i]);
        }
    }
    printf("\nOutputs after forward pass:\n");
    for (int j = 0; j < layer.neurons_amount; ++j) {
        printf("output[%d] = %f\n", j, layer.outputs[j]);
    }

    printf("Weights initialized using Xavier:\n");
    for (int j = 0; j < layer2.neurons_amount; ++j) {
        for (int i = 0; i < layer2.inputs_amount; ++i) {
            printf("W[%d][%d] = %f\n", j, i, layer2.weights[j * inputs + i]);
        }
    }
    printf("\nOutputs after forward pass:\n");
    for (int j = 0; j < layer2.neurons_amount; ++j) {
        printf("output[%d] = %f\n", j, layer2.outputs[j]);
    }

    const double loss = mse_loss(layer.outputs, target, layer.neurons_amount);

    printf("MSE loss: %f\n", loss);

    free_feed_forward(&layer);
    free_feed_forward(&layer2);
    return 0;
}
