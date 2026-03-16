#include "raylib.h"
#include "Visualizer.h"
#include "../../Entities/Layers/Linear/LinearLayer.h"
#include "../../Entities/Layers/Convolution/Conv2d.h"
#include "../../Entities/Layers/Linear/Dropout.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define UI_SIDEBAR_WIDTH 300
#define NODE_RADIUS 10
#define LAYER_WIDTH 180
#define LAYER_HEIGHT 220
#define LAYER_SPACING 300

#ifndef CYAN
    #define CYAN CLITERAL(Color){ 0, 255, 255, 255 }
#endif
#ifndef RAYWHITE
    #define RAYWHITE CLITERAL(Color){ 245, 245, 245, 255 }
#endif

// --- Internal Helpers ---

static void DrawGridCustom(int size, float spacing, Color color) {
    for (int i = -size; i <= size; i++) {
        DrawLineV((Vector2){ (float)i * spacing, (float)-size * spacing }, (Vector2){ (float)i * spacing, (float)size * spacing }, color);
        DrawLineV((Vector2){ (float)-size * spacing, (float)i * spacing }, (Vector2){ (float)size * spacing, (float)i * spacing }, color);
    }
}

static void DrawActivationCurve(double (*func)(double), Rectangle rect, Color color) {
    DrawRectangleLinesEx(rect, 1, Fade(color, 0.3f));
    Vector2 last_pt = {0};
    for (int i = 0; i < rect.width; i++) {
        float x_norm = (float)i / rect.width;
        double x_val = (x_norm - 0.5) * 10.0; // range [-5, 5]
        double y_val = func(x_val);
        
        float py = rect.y + rect.height - (float)((y_val + 2.0) / 4.0 * rect.height); // range [-2, 2] mapped to Rect
        py = fminf(fmaxf(py, rect.y), rect.y + rect.height);
        
        Vector2 pt = { rect.x + i, py };
        if (i > 0) DrawLineV(last_pt, pt, color);
        last_pt = pt;
    }
}

static void DrawLayerFrame(int x, int y, const char* name, NcLayerType type, bool selected, bool hovered) {
    Rectangle rec = { (float)x, (float)y, LAYER_WIDTH, LAYER_HEIGHT };
    
    // Glassmorphism effect
    DrawRectangleRec(rec, Fade(selected ? DARKBLUE : (hovered ? DARKGRAY : BLACK), 0.6f));
    DrawRectangleLinesEx(rec, selected ? 2 : 1, selected ? CYAN : (hovered ? RAYWHITE : GRAY));
    
    // Header
    DrawRectangle(x, y, LAYER_WIDTH, 30, Fade(GRAY, 0.3f));
    DrawText(name, x + 10, y + 8, 15, RAYWHITE);
}

static void DrawLinearModule(LinearLayer *ll, int x, int y) {
    int neurons = (ll->neurons_amount > 10) ? 10 : ll->neurons_amount;
    int inputs = (ll->inputs_amount > 10) ? 10 : ll->inputs_amount;
    
    for (int j = 0; j < neurons; j++) {
        int ny = y + 50 + j * 15;
        DrawCircle(x + 20, ny, 5, DARKGRAY);
        DrawCircleLines(x + 20, ny, 5, CYAN);
        
        if (ll->neurons_amount > 10 && j == 9) DrawText("...", x + 15, ny + 10, 10, GRAY);
    }
    
    DrawText(TextFormat("Neurons: %d", ll->neurons_amount), x + 40, y + 50, 10, GRAY);
    DrawText(TextFormat("Inputs: %d", ll->inputs_amount), x + 40, y + 65, 10, GRAY);
    
    // Weight Visualization (Heatmap placeholder)
    Rectangle heat = { (float)x + 20, (float)y + 100, 140, 100 };
    DrawRectangleLinesEx(heat, 1, DARKGRAY);
    DrawText("Weight Distribution", x + 25, y + 105, 10, Fade(RAYWHITE, 0.5f));
    
    if (ll->weights) {
        int grid_size = 8;
        float cw = heat.width / grid_size;
        float ch = heat.height / grid_size;
        for(int r=0; r<grid_size; r++) {
            for(int c=0; c<grid_size; c++) {
                double val = ((double*)ll->weights->data)[(r*grid_size + c) % ll->weights->size];
                Color col = (val > 0) ? Fade(BLUE, (float)val) : Fade(RED, (float)-val);
                DrawRectangleV((Vector2){heat.x + c*cw, heat.y + 20 + r*ch}, (Vector2){cw-1, ch-1}, col);
            }
        }
    }
}

// --- Public API ---

void visualizer_init(int width, int height) {
    // raylib should be initialized in main to control life cycle
}

void visualizer_run(NcModel *model, VisState *state) {
    // --- Input Handling ---
    if (IsMouseButtonDown(1)) {
        Vector2 delta = GetMouseDelta();
        state->camera.target = GetScreenToWorld2D((Vector2){
            state->camera.offset.x - delta.x,
            state->camera.offset.y - delta.y
        }, state->camera);
    }
    
    float wheel = GetMouseWheelMove();
    if (wheel != 0) {
        Vector2 mouseWorldPos = GetScreenToWorld2D(GetMousePosition(), state->camera);
        state->camera.offset = GetMousePosition();
        state->camera.target = mouseWorldPos;
        state->camera.zoom += wheel * 0.1f;
        if (state->camera.zoom < 0.1f) state->camera.zoom = 0.1f;
    }

    BeginDrawing();
    ClearBackground((Color){ 10, 10, 12, 255 });

    BeginMode2D(state->camera);
        DrawGridCustom(100, 50, Fade(GRAY, 0.1f));
        
        for (int i = 0; i < model->count; i++) {
            Layer *l = model->layers[i];
            int lx = i * LAYER_SPACING;
            int ly = 0;
            
            // Interaction Check
            Vector2 mouse = GetScreenToWorld2D(GetMousePosition(), state->camera);
            Rectangle lrec = { (float)lx, (float)ly, LAYER_WIDTH, LAYER_HEIGHT };
            bool hovered = CheckCollisionPointRec(mouse, lrec);
            if (hovered) state->hovered_layer = i;
            if (hovered && IsMouseButtonPressed(0)) state->selected_layer = i;
            
            DrawLayerFrame(lx, ly, l->name, l->type, state->selected_layer == i, hovered);
            
            // Specialized Drawing
            switch (l->type) {
                case NC_LAYER_LINEAR:
                    DrawLinearModule((LinearLayer*)l->layer_data, lx, ly);
                    break;
                    
                case NC_LAYER_ACTIVATION: {
                    // We need a way to get the function pointer. For now show "Curve"
                    DrawText("Activation Function", lx + 10, ly + 40, 10, GRAY);
                    DrawRectangleLines(lx + 10, ly + 60, 160, 100, DARKGRAY);
                    DrawCircle(lx + 80, ly + 110, 30, Fade(CYAN, 0.2f));
                    break;
                }
                
                case NC_LAYER_DROPOUT: {
                    DropoutLayer *dl = (DropoutLayer*)l->layer_data;
                    DrawText(TextFormat("Rate: %.2f", dl->dropout_rate), lx + 10, ly + 50, 15, GOLD);
                    DrawCircle(lx + 90, ly + 130, 40, Fade(GOLD, 0.1f));
                    DrawCircleLines(lx + 90, ly + 130, 40, GOLD);
                    break;
                }
                
                case NC_LAYER_CONV2D: {
                    Conv2dLayer *cl = (Conv2dLayer*)l->layer_data;
                    DrawText(TextFormat("Channels: %d->%d", cl->in_channels, cl->out_channels), lx+10, ly+40, 10, RAYWHITE);
                    DrawText(TextFormat("Kernel: %dx%d", cl->kernel_h, cl->kernel_w), lx+10, ly+55, 10, GRAY);
                    // Volume visualization
                    for(int v=0; v<3; v++) {
                        DrawRectangleLines(lx+40+v*10, ly+80+v*10, 80, 80, Fade(CYAN, 0.8f - v*0.2f));
                    }
                    break;
                }
                
                default:
                    DrawText("Static Layer", lx + 10, ly + 50, 12, GRAY);
                    break;
            }
            
            // Connections
            if (i < model->count - 1) {
                Vector2 start = { (float)lx + LAYER_WIDTH, (float)ly + 110 };
                Vector2 end = { (float)lx + LAYER_SPACING, (float)ly + 110 };
                DrawLineBezier(start, end, 2.0f, Fade(CYAN, 0.5f));
            }
        }
    EndMode2D();

    // --- HUD / Overlays ---
    
    // Header
    DrawRectangle(0, 0, GetScreenWidth(), 60, Fade(BLACK, 0.8f));
    DrawText("NEURAL-C ENGINE", 20, 15, 30, CYAN);
    DrawText(state->is_training ? "TRAINING ACTIVE" : "ENGINE IDLE", 350, 22, 18, state->is_training ? GOLD : LIME);

    // Loss Panel
    int lp_x = GetScreenWidth() - 320;
    DrawRectangle(lp_x, 80, 300, 200, Fade(BLACK, 0.6f));
    DrawRectangleLines(lp_x, 80, 300, 200, GRAY);
    DrawText("LOSS HISTORY", lp_x + 10, 90, 15, RAYWHITE);
    
    if (state->history_count > 1) {
        for (int i = 0; i < state->history_count - 1; i++) {
            float x1 = lp_x + 10 + ((float)i / state->history_capacity) * 280;
            float y1 = 260 - (state->loss_history[i] * 100);
            float x2 = lp_x + 10 + ((float)(i + 1) / state->history_capacity) * 280;
            float y2 = 260 - (state->loss_history[i+1] * 100);
            DrawLineV((Vector2){x1, fminf(fmaxf(y1, 100), 270)}, (Vector2){x2, fminf(fmaxf(y2, 100), 270)}, GREEN);
        }
    }
    
    // Details Panel (Sidebar)
    if (state->selected_layer != -1) {
        Layer *sl = model->layers[state->selected_layer];
        DrawRectangle(0, 60, UI_SIDEBAR_WIDTH, GetScreenHeight() - 60, Fade(BLACK, 0.9f));
        DrawLine(UI_SIDEBAR_WIDTH, 60, UI_SIDEBAR_WIDTH, GetScreenHeight(), GRAY);
        
        DrawText("LAYER INSPECTOR", 20, 80, 20, CYAN);
        DrawText(TextFormat("Name: %s", sl->name), 20, 115, 15, RAYWHITE);
        
        const char* type_str = "Unknown";
        if(sl->type == NC_LAYER_LINEAR) type_str = "Linear / Dense";
        else if(sl->type == NC_LAYER_CONV2D) type_str = "Convolutional 2D";
        else if(sl->type == NC_LAYER_ACTIVATION) type_str = "Activation";
        else if(sl->type == NC_LAYER_DROPOUT) type_str = "Dropout";
        
        DrawText(TextFormat("Type: %s", type_str), 20, 140, 15, GRAY);
        
        int p_count = 0;
        if (sl->get_parameters) {
            Tensor **p = sl->get_parameters(sl, &p_count);
            DrawText(TextFormat("Parameters: %d Tensors", p_count), 20, 170, 15, RAYWHITE);
            for(int j=0; j<p_count; j++) {
                DrawText(TextFormat("  P%d: [%d, %d]", j, p[j]->shape[0], p[j]->shape[1]), 20, 195 + j*20, 12, GRAY);
            }
            free(p);
        }
    }
    
    // Help HUD
    DrawText("RMB: Pan | Scroll: Zoom | Click: Inspect", 20, GetScreenHeight() - 30, 15, DARKGRAY);

    EndDrawing();
}

void visualizer_add_loss(VisState *state, float loss) {
    if (state->history_count < state->history_capacity) {
        state->loss_history[state->history_count++] = loss;
    } else {
        // Shift left
        for (int i = 0; i < state->history_capacity - 1; i++) {
            state->loss_history[i] = state->loss_history[i+1];
        }
        state->loss_history[state->history_capacity - 1] = loss;
    }
    state->current_loss = loss;
}