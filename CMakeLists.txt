cmake_minimum_required(VERSION 4.0)
project(NeuralC C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralC main.c
        Utils/ActivationFunctions/ActivationFunctions.c
        Utils/ActivationFunctions/ActivationFunctions.h
        Entities/Layers/FeedForward/FeedForward.c
        Entities/Layers/FeedForward/FeedForward.h
        Utils/Optimizers/Optimizers.c
        Utils/Initializers/Initializers.c
        Utils/Initializers/Initializers.h)
