#include "StructuresInitializers.h"
#include "../RandomInitializers/RandomInitializers.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

void orthogonal_init(double *w, const int fan_in, const int fan_out, const int rows, const int cols,
                     const double gain) {
    const int size = rows * cols;
    double *A = malloc(sizeof(double) * size);
    if (!A) return;

    for (int i = 0; i < size; ++i)
        A[i] = random_normal_initializer(0.0, 1.0);

    for (int j = 0; j < cols; ++j) {
        double norm = 0.0;
        for (int i = 0; i < rows; ++i)
            norm += A[i * cols + j] * A[i * cols + j];

        norm = sqrt(norm);
        if (norm < 1e-8) continue;

        for (int i = 0; i < rows; ++i)
            A[i * cols + j] /= norm;

        for (int k = j + 1; k < cols; ++k) {
            double dot = 0.0;
            for (int i = 0; i < rows; ++i)
                dot += A[i * cols + j] * A[i * cols + k];
            for (int i = 0; i < rows; ++i)
                A[i * cols + k] -= dot * A[i * cols + j];
        }
    }

    for (int i = 0; i < size; ++i)
        w[i] = A[i] * gain;

    free(A);
}

void sparse_init(double *w, const int fan_in, const int fan_out, const int rows, const int cols,
                 const int nonezeros_per_row, const double std) {
    memset(w, 0, sizeof(double) * rows * cols);

    for (int r = 0; r < rows; ++r) {
        for (int k = 0; k < nonezeros_per_row; ++k) {
            const int c = rand() % cols;
            w[r * cols + c] = random_normal_initializer(0.0, std);
        }
    }
}

void constant_init(double *w, const int rows, const int cols, const double value) {
    for (int i = 0; i < rows * cols; ++i)
        w[i] = value;
}

void dirac_init(double *w, const int rows, const int cols, const double gain) {
    memset(w, 0, sizeof(double) * rows * cols);
    const int k = rows < cols ? rows : cols;
    for (int i = 0; i < k; ++i)
        w[i * cols + i] = gain;
}

void delta_orthogonal_init(double *w, const int rows, const int cols, const double gain) {
    memset(w, 0, sizeof(double) * rows * cols);

    const int k = rows < cols ? rows : cols;
    double *temp = malloc(sizeof(double) * rows * cols);
    if (!temp) return;

    for (int i = 0; i < rows * cols; ++i)
        temp[i] = random_normal_initializer(0.0, 1.0);

    for (int j = 0; j < k; ++j) {
        double norm = 0.0;
        for (int i = 0; i < rows; ++i)
            norm += temp[i * cols + j] * temp[i * cols + j];

        norm = sqrt(norm);
        if (norm < 1e-12) continue;

        for (int i = 0; i < rows; ++i)
            temp[i * cols + j] /= norm;
    }

    for (int i = 0; i < k; ++i)
        w[i * cols + i] = temp[i * cols + i] * gain;

    free(temp);
}

void block_diagonal_init(double *w, const int rows, const int cols, const double gain) {
    memset(w, 0, sizeof(double) * rows * cols);
    const int k = rows < cols ? rows : cols;
    for (int i = 0; i < k; ++i)
        w[i * cols + i] = gain;
}
