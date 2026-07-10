$ErrorActionPreference = "Stop"

$BuildDir = "build"
$ExecutableName = "NeuralC"

if (Test-Path $BuildDir) {
    Write-Host "Cleaning build directory..." -ForegroundColor Cyan
    Remove-Item -Recurse -Force $BuildDir
}

New-Item -ItemType Directory -Path $BuildDir | Out-Null
Write-Host "Created clean build directory." -ForegroundColor Cyan

Write-Host "Configuring project with CMake..." -ForegroundColor Cyan
cmake -S . -B $BuildDir -G "MinGW Makefiles"

Write-Host "Building the project..." -ForegroundColor Cyan
cmake --build $BuildDir --config Release

Write-Host "Running $ExecutableName...`n" -ForegroundColor Green

$ExePath = "$BuildDir/$ExecutableName.exe"
$ConfigExePath = "$BuildDir/Release/$ExecutableName.exe"

if (Test-Path $ConfigExePath) {
    & $ConfigExePath
} elseif (Test-Path $ExePath) {
    & $ExePath
} else {
    Write-Error "Executable not found! Check your build output."
}