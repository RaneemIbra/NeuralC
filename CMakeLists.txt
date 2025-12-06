cmake_minimum_required(VERSION 4.0)
project(NeuralC C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralC main.c
        Entities/AtomicNeuron/AtomicNeuron.c
        Entities/AtomicNeuron/AtomicNeuron.h
        Utils/ActivationFunctions/ActivationFunctions.c
        Utils/ActivationFunctions/ActivationFunctions.h
        Entities/Layers/FeedForward/FeedForward.c
        Entities/Layers/FeedForward/FeedForward.h)
