#ifndef NEURALC_TRAINING_H
#define NEURALC_TRAINING_H

#include "../Entities/Model/Model.h"

void perform_epochs(Model *model, void *optimizer, bool is_adam, Tensor *x, Tensor *y_true,
                    double (*loss_fn)(const Tensor*, const Tensor*),
                    NcStatus (*loss_derivative)(const Tensor*, const Tensor*, Tensor**),
                    int epochs);

#endif //NEURALC_TRAINING_H