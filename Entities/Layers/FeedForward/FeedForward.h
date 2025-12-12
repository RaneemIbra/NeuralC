#ifndef NEURALC_FEEDFORWARD_H
#define NEURALC_FEEDFORWARD_H

typedef void (*WeightsInitializer)(double* weights, int fan_in, int fan_out, int rows, int cols);

typedef struct {
    int inputs_amount;
    int neurons_amount;
    double* weights;
    double* biases;
    double* outputs;
    double (*activation_function)(double);
} FeedForward;

void init_feed_forward(FeedForward* layer, int inputs_amount ,int neurons_amount, double (*activation_function)(double), WeightsInitializer weights_initializer);
void free_feed_forward(FeedForward* layer);
void forward_pass(const FeedForward* layer, const double* inputs);

#endif //NEURALC_FEEDFORWARD_H