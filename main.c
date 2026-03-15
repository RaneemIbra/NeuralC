#include <stdio.h>
#include <stdlib.h>
#include "Entities/Model/Model.h"
#include "Entities/Layers/FeedForward/FeedForward.h"
#include "Entities/Layers/ActivationLayer.h"
#include "Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h"
#include "Utils/Initializers/Initializers.h"
#include "Core/Tensor.h"
#include "Core/Common.h"

int main(void) {
    printf("=== NeuralC: Professional Refined Engine ===\n");

    // 1. Create a Sequential Model
    Model *model = model_create();

    // 2. Add Layers
    model_add(model, dense_layer_create(3, 4, NULL, "dense1", get_initializer(INIT_HE_NORMAL)));
    model_add(model, activation_layer_create(relu_activation, "relu1"));
    model_add(model, dense_layer_create(4, 2, NULL, "dense2", get_initializer(INIT_XAVIER_NORMAL)));
    model_add(model, activation_layer_create(sigmoid_activation, "sigmoid1"));

    printf("Model initialized with %d layers.\n", model->count);

    // 3. Create Input Tensor
    int in_shape[] = {3, 1};
    Tensor *input = tensor_create(DTYPE_FLOAT64, in_shape, 2, false);
    if (!input) {
        NC_REPORT_ERROR(NC_ERROR_OUT_OF_MEMORY);
        return 1;
    }
    double *d = (double*)input->data;
    d[0] = 1.0; d[1] = 0.5; d[2] = -1.0;

    printf("\nInput Tensor:\n");
    tensor_print(input);

    // 4. Run Model Forward Pass with Error Handling
    Tensor *output = NULL;
    NcStatus status = model_forward(model, input, &output);

    if (status == NC_SUCCESS) {
        printf("\nFinal Model Output (Success):\n");
        tensor_print(output);
    } else {
        printf("\nModel execution failed: %s\n", nc_status_to_string(status));
    }

    // 5. Cleanup
    tensor_free(input);
    model_free(model);

    printf("\nRefinement test completed successfully!\n");
    return 0;
}
