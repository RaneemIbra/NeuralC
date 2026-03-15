#ifndef NEURALC_ACTIVATIONLAYER_H
#define NEURALC_ACTIVATIONLAYER_H

#include "Layer.h"

Layer* activation_layer_create(double (*activation)(double), double (*derivative)(double), const char* name);

#endif //NEURALC_ACTIVATIONLAYER_H
