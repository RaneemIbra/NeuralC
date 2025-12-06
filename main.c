#include <stdio.h>
#include "Utils/ActivationFunctions/ActivationFunctions.h"
#include "Entities/Layers/FeedForward/FeedForward.h"

int main(void) {
    FeedForward layer;
    int inputs = 3;
    int neurons = 2;

    init_feed_forward(&layer, inputs, neurons, relu_activation);

    layer.weights[0 * inputs + 0] = 0.5;
    layer.weights[0 * inputs + 1] = 0.5;
    layer.weights[0 * inputs + 2] = 0.0;
    layer.biases[0] = 0.1;

    layer.weights[1 * inputs + 0] = -1.0;
    layer.weights[1 * inputs + 1] = 2.0;
    layer.weights[1 * inputs + 2] = 0.5;
    layer.biases[1] = -0.2;

    double in[] = {1.0, 2.0, -1.0};
    forward_pass(&layer, in);

    for (int j = 0; j < layer.neurons_amount; ++j) {
        printf("output[%d] = %f\n", j, layer.outputs[j]);
    }

    free_feed_forward(&layer);
    return 0;
}