#include <stdio.h>
#include <stdlib.h>
#include "Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "Entities/Layers/FeedForward/FeedForward.h"
#include "Utils/Initializers/Initializers.h"
#include "Utils/LossFunctions/LossFunctions.h"
#include "Core/Tensor.h"

void example_single_layer() {
    printf("=== Example 1: Single Layer Forward Pass ===\n");
    FeedForward layer;
    init_feed_forward(&layer, 3, 2, relu_activation, "relu", get_initializer(INIT_XAVIER_UNIFORM));

    int in_shape[] = {3, 1};
    Tensor *input = tensor_create(DTYPE_FLOAT64, in_shape, 2, false);
    double *d = (double*)input->data;
    d[0] = 0.5; d[1] = -0.2; d[2] = 0.1;

    Tensor *output = forward_pass(&layer, input);
    tensor_print(output);

    tensor_free(input);
    free_feed_forward(&layer);
    printf("\n");
}

void example_multi_layer() {
    printf("=== Example 2: Multi-Layer (Deep) Network ===\n");
    // Layer 1: 3 -> 4
    FeedForward l1;
    init_feed_forward(&l1, 3, 4, relu_activation, "relu", get_initializer(INIT_HE_NORMAL));
    
    // Layer 2: 4 -> 2
    FeedForward l2;
    init_feed_forward(&l2, 4, 2, sigmoid_activation, "sigmoid", get_initializer(INIT_XAVIER_NORMAL));

    // Input: Vector of 3
    int in_shape[] = {3, 1};
    Tensor *input = tensor_zeros(DTYPE_FLOAT64, in_shape, 2, false);
    ((double*)input->data)[0] = 1.0; 

    // Propagate: Input -> L1 -> L2
    Tensor *hidden = forward_pass(&l1, input);
    Tensor *final_output = forward_pass(&l2, hidden);

    printf("Final Output (after 2 layers):\n");
    tensor_print(final_output);

    // Cleanup
    tensor_free(input);
    free_feed_forward(&l1);
    free_feed_forward(&l2);
    printf("\n");
}

void example_different_types() {
    printf("=== Example 3: Handling Different Data Types ===\n");
    // Note: Our current matmul focuses on FLOAT64, but the Tensor struct handles FLOAT32 too.
    int shape[] = {2, 2};
    Tensor *t_f32 = tensor_zeros(DTYPE_FLOAT32, shape, 2, false);
    Tensor *t_f64 = tensor_zeros(DTYPE_FLOAT64, shape, 2, false);

    printf("Float32 Tensor:\n");
    tensor_print(t_f32);
    printf("Float64 Tensor:\n");
    tensor_print(t_f64);

    tensor_free(t_f32);
    tensor_free(t_f64);
    printf("\n");
}

int main(void) {
    printf("NeuralC Tensor Usage Examples\n");
    printf("-----------------------------\n\n");

    example_single_layer();
    example_multi_layer();
    example_different_types();

    return 0;
}
