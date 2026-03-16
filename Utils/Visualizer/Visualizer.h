#ifndef NEURALC_VISUALIZER_H
#define NEURALC_VISUALIZER_H

#include "../../Entities/Model/Model.h"
#include "raylib.h"

typedef struct {
    float *loss_history;
    int history_count;
    int history_capacity;
    bool is_training;
    
    // Camera and Interactivity
    Camera2D camera;
    int selected_layer;
    int hovered_layer;
    
    // Performance
    float current_loss;
    double training_speed; // items/sec
} VisState;

void visualizer_init(int width, int height);
void visualizer_run(NcModel *model, VisState *state);
void visualizer_add_loss(VisState *state, float loss);

#endif