cmake_minimum_required(VERSION 4.0)
project(NeuralC C)

add_compile_definitions(_USE_MATH_DEFINES)

set(CMAKE_C_STANDARD 11)

add_executable(NeuralC main.c
        Core/Tensor.c
        Core/Tensor.h
        Core/TensorMath.c
        Core/TensorMath.h
        Core/Common.h
        Core/Graph.c
        Core/Graph.h
        Core/Training.h
        Core/Training.c
        Demos/CNN_demos.c
        Demos/CNN_demos.h
        Demos/FFN_demos.c
        Demos/FFN_demos.h
        Entities/Model/Model.c
        Entities/Model/Model.h
        Entities/Layers/Activations/ActivationLayer.c
        Entities/Layers/Activations/ActivationLayer.h
        Entities/Layers/Linear/LinearLayer.c
        Entities/Layers/Linear/LinearLayer.h
        Entities/Layers/Linear/Flatten.c
        Entities/Layers/Linear/Flatten.h
        Entities/Layers/Linear/Dropout.c
        Entities/Layers/Linear/Dropout.h
        Entities/Layers/Convolution/Conv2d.c
        Entities/Layers/Convolution/Conv2d.h
        Entities/Layers/Pooling/Pooling.c
        Entities/Layers/Pooling/Pooling.h
        Entities/Layers/Layer.h
        Utils/Optimizers/SGD/SGD.c
        Utils/Optimizers/SGD/SGD.h
        Utils/Optimizers/Adam/Adam.c
        Utils/Optimizers/Adam/Adam.h
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
        Utils/Initializers/BiasInitializers/BiasInitializers.h
        Utils/LossFunctions/LossFunctions.h
        Utils/LossFunctions/RegressionsLosses.c
        Utils/LossFunctions/ClassificationLosses.c
        Utils/LossFunctions/ProbabilisticLosses.c)
