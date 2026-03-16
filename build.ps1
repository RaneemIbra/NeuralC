$gccPath = "C:\raylib\w64devkit\bin\gcc.exe"
$binPath = "C:\raylib\w64devkit\bin"
$raylibPath = "C:\raylib\w64devkit\x86_64-w64-mingw32"
$includePath = "-I$raylibPath\include -I. -std=c99"
$libPath = "-L$raylibPath\lib"
$libs = "-lraylib -lopengl32 -lgdi32 -lwinmm -lm"

$srcFiles = @(
    "main.c",
    "Core/Tensor.c",
    "Core/TensorMath.c",
    "Core/Graph.c",
    "Entities/Model/Model.c",
    "Entities/Layers/Activations/ActivationLayer.c",
    "Entities/Layers/Linear/LinearLayer.c",
    "Entities/Layers/Linear/Flatten.c",
    "Entities/Layers/Linear/Dropout.c",
    "Entities/Layers/Convolution/Conv2d.c",
    "Entities/Layers/Pooling/Pooling.c",
    "Utils/Optimizers/SGD/SGD.c",
    "Utils/Optimizers/Adam/Adam.c",
    "Utils/Initializers/Initializers.c",
    "Utils/ActivationFunctions/ParamActivations/ParamActivations.c",
    "Utils/ActivationFunctions/ScalarActivations/ScalarActivations.c",
    "Utils/ActivationFunctions/VectorActivations/VectorActivations.c",
    "Utils/Initializers/RandomInitializers/RandomInitializers.c",
    "Utils/Initializers/VarianceInitializers/VarianceInitializers.c",
    "Utils/Initializers/StructuresInitializers/StructuresInitializers.c",
    "Utils/Initializers/BiasInitializers/BiasInitializers.c",
    "Utils/LossFunctions/RegressionsLosses.c",
    "Utils/LossFunctions/ClassificationLosses.c",
    "Utils/LossFunctions/ProbabilisticLosses.c",
    "Utils/Visualizer/Visualizer.c"
)

# Update PATH to prioritize w64devkit
$env:PATH = "$binPath;$env:PATH"

$cmd = "$gccPath -o NeuralC $($srcFiles -join ' ') $includePath $libPath $libs"
Write-Host "Compiling..."
Invoke-Expression $cmd
if ($LASTEXITCODE -eq 0) {
    Write-Host "Compilation successful! Running NeuralC..."
    .\NeuralC.exe
} else {
    Write-Host "Compilation failed."
}
