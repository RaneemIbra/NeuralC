#include "FeedForward.h"

#include <stddef.h>
#include <stdlib.h>

#include "../../AtomicNeuron/AtomicNeuron.h"

void init_feed_forward(FeedForward* layer, const int neurons_amount, double (*activation_function)(double)) {
    if (neurons_amount < 1) {
        layer->neurons_amount = 0;
        layer->neurons = NULL;
        return;
    }
    layer->neurons_amount = neurons_amount;
    layer->neurons = (AtomicNeuron*)malloc(neurons_amount * sizeof(AtomicNeuron));
    if (layer->neurons == NULL) {
        return;
    }

    for (int i = 0; i < neurons_amount; i++) {
        init_atomic_neuron(&layer->neurons[i], 0.0, activation_function);
    }
}

void free_feed_forward(FeedForward* layer) {
    if (layer != NULL && layer->neurons != NULL) {
        free(layer->neurons);
        layer->neurons = NULL;
        layer->neurons_amount = 0;
    }
}

void forward_pass(FeedForward* layer) {
}