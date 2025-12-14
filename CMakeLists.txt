cmake_minimum_required(VERSION 4.0)
project(NeuralC C)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralC main.c
        Entities/Layers/FeedForward/FeedForward.c
        Entities/Layers/FeedForward/FeedForward.h
        Utils/Optimizers/Optimizers.c
        Utils/Optimizers/Optimizers.h
        Utils/Initializers/Initializers.c
        Utils/Initializers/Initializers.h
        Utils/ActivationFunctions/ParamActivations/ParamActivations.c
        Utils/ActivationFunctions/ParamActivations/ParamActivations.h
        Utils/ActivationFunctions/ScalarActivations/ScalarActivations.c
        Utils/ActivationFunctions/ScalarActivations/ScalarActivations.h
        Utils/ActivationFunctions/VectorActivations/VectorActivations.c
        Utils/ActivationFunctions/VectorActivations/VectorActivations.h
        Utils/Initializers/RandomInitializers/RandomInitializers.c
        Utils/Initializers/RandomInitializers/RandomInitializers.h
        Utils/Initializers/VarianceInitializers/VarianceInitializers.c
        Utils/Initializers/VarianceInitializers/VarianceInitializers.h
        Utils/Initializers/StructuresInitializers/StructuresInitializers.c
        Utils/Initializers/StructuresInitializers/StructuresInitializers.h
        Utils/Initializers/BiasInitializers/BiasInitializers.c
        Utils/Initializers/BiasInitializers/BiasInitializers.h)
