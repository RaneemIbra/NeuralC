#include "../Entities/Model/Model.h"
#include "../Utils/Optimizers/SGD/SGD.h"
#include "../Utils/Optimizers/Adam/Adam.h"

void perform_epochs(Model *model, void *optimizer, bool is_adam, Tensor *x, Tensor *y_true,
                    double (*loss_fn)(const Tensor*, const Tensor*),
                    NcStatus (*loss_derivative)(const Tensor*, const Tensor*, Tensor**),
                    int epochs) {
    for (int epoch = 1; epoch <= epochs; epoch++) {
        Tensor *y_pred = NULL;
        model_forward(model, x, &y_pred);
        double loss = loss_fn(y_pred, y_true);
        printf("    Epoch %d - Loss: %.6f\n", epoch, loss);

        Tensor *grad_output = NULL;
        loss_derivative(y_pred, y_true, &grad_output);

        NcGraph *backprop_graph = nc_graph_create();
        model_backward(model, grad_output, backprop_graph);

        if (is_adam) {
            optimizer_adam_step((AdamOptimizer*)optimizer);
        } else {
            optimizer_sgd_step((SGDOptimizer*)optimizer);
        }

        nc_graph_free(backprop_graph);
        tensor_free(grad_output);
    }
}