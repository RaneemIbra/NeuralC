#ifndef NEURALC_VARIANCEINITIALIZERS_H
#define NEURALC_VARIANCEINITIALIZERS_H

#include "../../../Core/Tensor.h"

void xavier_uniform_initializer(Tensor *w, int fan_in, int fan_out);

void xavier_normal_initializer(Tensor *w, int fan_in, int fan_out);

void he_uniform_initializer(Tensor *w, int fan_in, int fan_out);

void he_normal_initializer(Tensor *w, int fan_in, int fan_out);

void lecun_uniform_initializer(Tensor *w, int fan_in, int fan_out);

void lecun_normal_initializer(Tensor *w, int fan_in, int fan_out);

void variance_scaling(Tensor *w, int fan_in, int fan_out, double scale, int mode, int uniform);

void truncated_normal(Tensor *w, int fan_in, int fan_out, double gain);

#endif //NEURALC_VARIANCEINITIALIZERS_H
