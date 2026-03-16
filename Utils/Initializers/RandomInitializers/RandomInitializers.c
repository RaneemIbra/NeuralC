#include "RandomInitializers.h"
#include <math.h>
#include <stdlib.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif

double random_uniform_initializer(const double min, const double max) {
    return min + (double) rand() / RAND_MAX * (max - min);
}

double random_normal_initializer(const double mean, const double std) {
    const double u1 = ((double) rand() + 1) / ((double) RAND_MAX + 1);
    const double u2 = ((double) rand() + 1) / ((double) RAND_MAX + 1);
    const double z = sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
    return mean + z * std;
}

double random_truncated_normal_initializer(const double std) {
    double x;
    do {
        x = random_normal_initializer(0.0, std);
    } while (fabs(x) > 2.0 * std);
    return x;
}
