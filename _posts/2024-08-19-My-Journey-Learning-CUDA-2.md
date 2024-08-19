---
layout: post
title: My Journey Learning CUDA 2
categories: CUDA
tags: [CUDA]
---

# Introduction

CUDA (Compute Unified Device Architecture) is a parallel computing platform and application programming interface (API) model created by NVIDIA. It allows developers to use a CUDA-enabled graphics processing unit (GPU) for general purpose processing – an approach known as GPGPU (General-Purpose computing on Graphics Processing Units). In this blog post, we will guide you through the process of setting up the CUDA development environment and running a simple "Hello World" program.

# Setting Up the CUDA Development Environment on Windows
To set up the CUDA development environment on Windows, follow these steps:

1. Install the CUDA Toolkit:
   - Visit the NVIDIA official website (https://developer.nvidia.com/cuda-toolkit-archive) and download the CUDA Toolkit that matches your operating system and hardware requirements.
   - Follow the installation instructions provided by NVIDIA to install the CUDA Toolkit on your system.
   - Add the CUDA Toolkit binaries to your system's PATH environment variable.

2. Verify Installation:
   - Open a terminal or command prompt and type `nvcc --version` to verify that the CUDA Toolkit is installed correctly.

3. Install Visual Studio:
   - CUDA requires a compatible version of Visual Studio to compile CUDA programs.
   - Download and install Visual Studio from the official Microsoft website (https://visualstudio.microsoft.com/).
   - During the installation process, select the "Desktop development with C++" workload to install the necessary components for CUDA development.

4. Configure Visual Studio for CUDA:
   - Open Visual Studio and select "File" > "New" > "Project".
   - In the "New Project" window, select "Visual C++" > "Empty Project" and give your project a name.
   - Click "OK" to create the project.
   - Right-click on the project in the Solution Explorer and select "Properties".
   - In the "Properties" window, navigate to "CUDA C/C++" > "Device" > "Code Generation".
   - Set the "Compute" and "SM" (Streaming Multiprocessor) versions to match your GPU's capabilities.
   - Click "Apply" and then "OK" to save the changes.

5. Add CUDA Libraries to the Project:
   - Right-click on the project in the Solution Explorer and select "Properties".
   - In the "Properties" window, navigate to "VC++ Directories".
   - Add the CUDA Toolkit libraries to the "Include Directories" and "Library Directories" fields.
   - The paths should be something like "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\include" and "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.X\lib\x64", respectively, where "vX.X" represents the version of the CUDA Toolkit you have installed.

6. Link CUDA Libraries:
   - In the "Properties" window, navigate to "Linker" > "Input".
   - Add the CUDA libraries to the "Additional Dependencies" field.
   - The libraries should be something like "cudart.lib", "cublas.lib", "curand.lib", etc., depending on the libraries you are using.

7. Verify Setup:
   - Create a new CUDA source file in your project and write a simple CUDA program.
   - Compile and run the program to verify that the CUDA development environment is set up correctly.

That's it! You have successfully set up the CUDA development environment on your Windows system. You can now start developing CUDA applications and leveraging the power of GPUs for general-purpose processing.

reference: https://docs.nvidia.com/cuda/


#  Running a Hello World Program:

1. Create a new CUDA project in Visual Studio.
2. Add a new CUDA source file to the project.
3. Compile and run the program to verify that it works correctly.

Here is an example of a simple CUDA program that prints "Hello World" to the console:

```cpp
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



```

{% raw %}
```cpp
int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
```
{% endraw %}