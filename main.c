#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "Demos/FFN_demos.h"
#include "Demos/CNN_demos.h"

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
