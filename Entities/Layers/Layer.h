#ifndef NEURALC_LAYER_H
#define NEURALC_LAYER_H

#include "../../Core/Tensor.h"
#include "../../Core/Common.h"
#include "../../Core/Graph.h"

/**
 * Generic Layer Interface
 */
typedef struct Layer {
    char *name;
    void *layer_data; 
    bool train_mode;

    // forward now takes an optional NcGraph for intermediate memory management
    NcStatus (*forward)(struct Layer *self, int num_inputs, const Tensor **inputs, Tensor **output, NcGraph *graph);
    // backward pass: receives grad_output, returns grad_input (optional NcGraph for intermediates)
    NcStatus (*backward)(struct Layer *self, const Tensor *grad_output, Tensor **grad_input, NcGraph *graph);
    
    void (*reset_state)(struct Layer *self);
    void (*toggle_train)(struct Layer *self, bool train);

    Tensor** (*get_parameters)(struct Layer *self, int *num_params);
    void (*save)(struct Layer *self, const char *path);

    void (*free)(struct Layer *self);
} Layer;

/* Helper for 1-input layers */
static inline NcStatus layer_forward_1(Layer *l, const Tensor *in, Tensor **out, NcGraph *g) {
    if (!l || !l->forward) return NC_ERROR_NULL_POINTER;
    return l->forward(l, 1, &in, out, g);
}

#endif //NEURALC_LAYER_H
