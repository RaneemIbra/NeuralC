#ifndef NEURALC_STRUCTURED_INIT_H
#define NEURALC_STRUCTURED_INIT_H

void orthogonal_init(double* w, int fan_in, int fan_out, int rows, int cols, double gain);
void sparse_init(double* w, int fan_in, int fan_out, int rows, int cols, int nonezeros_per_row, double std);
void constant_init(double* w, int rows, int cols, double value);
void dirac_init(double* w, int rows, int cols, double gain);
void delta_orthogonal_init(double* w, int rows, int cols, double gain);
void block_diagonal_init(double* w, int rows, int cols, double gain);

#endif /* NEURALC_STRUCTURED_INIT_H */
