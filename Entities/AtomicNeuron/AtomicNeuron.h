#ifndef NEURALC_ATOMICNEURON_H
#define NEURALC_ATOMICNEURON_H

typedef struct {
    double input;
    double output;
}AtomicNeuron;

void init_atomic_neuron(AtomicNeuron* neuron, double input);

#endif //NEURALC_ATOMICNEURON_H