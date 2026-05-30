#!/bin/bash

set -e

BuildDir="build"
ExecutableName="NeuralC"

echo -e "\033[36mCleaning build directory...\033[0m"
if [ -d "$BuildDir" ]; then
    rm -rf "$BuildDir"
fi

mkdir -p "$BuildDir"
echo -e "\033[36mCreated clean build directory.\033[0m"

echo -e "\033[36mConfiguring project with CMake...\033[0m"
cmake -S . -B "$BuildDir"

echo -e "\033[36mBuilding the project...\033[0m"
cmake --build "$BuildDir" --config Release

echo -e "\033[32mRunning $ExecutableName...\n\033[0m"

ExePath="$BuildDir/$ExecutableName"

if [ -f "$ExePath" ]; then
    "$ExePath"
else
    echo -e "\033[31mError: Executable not found! Check your build output.\033[0m"
    exit 1
fi
