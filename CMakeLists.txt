cmake_minimum_required(VERSION 3.20)
project(NeuralC C)

set(CMAKE_C_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(NeuralC 
    main.c
    Core/Tensor.c
    Core/TensorMath.c
    Core/Graph.c
    Core/Training.c
    Demos/CNN_demos.c
    Demos/FFN_demos.c
    Entities/Model/Model.c
    Entities/Layers/Activations/ActivationLayer.c
    Entities/Layers/Linear/LinearLayer.c
    Entities/Layers/Linear/Flatten.c
    Entities/Layers/Linear/Dropout.c
    Entities/Layers/Convolution/Conv2d.c
    Entities/Layers/Pooling/Pooling.c
    Utils/Optimizers/SGD/SGD.c
    Utils/Optimizers/Adam/Adam.c
    Utils/Initializers/Initializers.c
    Utils/ActivationFunctions/ParamActivations/ParamActivations.c
    Utils/ActivationFunctions/ScalarActivations/ScalarActivations.c
    Utils/ActivationFunctions/VectorActivations/VectorActivations.c
    Utils/Initializers/RandomInitializers/RandomInitializers.c
    Utils/Initializers/VarianceInitializers/VarianceInitializers.c
    Utils/Initializers/StructuresInitializers/StructuresInitializers.c
    Utils/Initializers/BiasInitializers/BiasInitializers.c
    Utils/LossFunctions/RegressionsLosses.c
    Utils/LossFunctions/ClassificationLosses.c
    Utils/LossFunctions/ProbabilisticLosses.c
)

target_include_directories(NeuralC PRIVATE 
    .
    Core
    Demos
    Entities/Model
    Entities/Layers
    Entities/Layers/Activations
    Entities/Layers/Linear
    Entities/Layers/Convolution
    Entities/Layers/Pooling
    Utils/Optimizers/SGD
    Utils/Optimizers/Adam
    Utils/Initializers
    Utils/ActivationFunctions/ParamActivations
    Utils/ActivationFunctions/ScalarActivations
    Utils/ActivationFunctions/VectorActivations
    Utils/Initializers/RandomInitializers
    Utils/Initializers/VarianceInitializers
    Utils/Initializers/StructuresInitializers
    Utils/Initializers/BiasInitializers
    Utils/LossFunctions
)

add_compile_definitions(_USE_MATH_DEFINES)

if(NOT WIN32)
    target_link_libraries(NeuralC m)
endif()