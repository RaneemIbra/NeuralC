#ifndef NEURALC_VARIANCEINITIALIZERS_H
#define NEURALC_VARIANCEINITIALIZERS_H

void xavier_uniform_initializer(double* w, int fan_in, int fan_out, int rows, int cols);
void xavier_normal_initializer(double* w, int fan_in, int fan_out, int rows, int cols);

void he_uniform_initializer(double* w, int fan_in, int fan_out, int rows, int cols);
void he_normal_initializer(double* w, int fan_in, int fan_out, int rows, int cols);

void lecun_uniform_initializer(double* w, int fan_in, int fan_out, int rows, int cols);
void lecun_normal_initializer(double* w, int fan_in, int fan_out, int rows, int cols);

void variance_scaling(double* w, int fan_in, int fan_out, int rows, int cols, double scale, int mode, int uniform);

void truncated_normal(double* w, int fan_in, int fan_out, int rows, int cols, double gain);

#endif //NEURALC_VARIANCEINITIALIZERS_H