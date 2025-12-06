#ifndef NEURALC_FEEDFORWARD_H
#define NEURALC_FEEDFORWARD_H
#include "../../AtomicNeuron/AtomicNeuron.h"

typedef struct {
    int neurons_amount;
    AtomicNeuron* neurons;

} FeedForward;

void init_feed_forward(FeedForward* layer, int neurons_amount, double (*activation_function)(double));
void free_feed_forward(FeedForward* layer);
void forward_pass(FeedForward* layer);

#endif //NEURALC_FEEDFORWARD_H