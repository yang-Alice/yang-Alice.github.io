---
layout: post
title: Getting Started with CUDA 
subtitle:  Setting Up the Development Environment and Running a Hello World Program
author: https://yang-alice.github.io/
categories: CUDA
banner:
  #video: https://vjs.zencdn.net/v/oceans.mp4
  loop: true
  volume: 0.8
  start_at: 8.5
  #image: /assets/images/leetcode/33/Figure_5.png
  opacity: 0.618
  background: "#000"
  height: "100vh"
  min_height: "38vh"
  heading_style: "font-size: 4.25em; font-weight: bold; text-decoration: underline"
  subheading_style: "color: gold"
tags: CUDA
sidebar: []
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

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}

```

result:
```
{1,2,3,4,5} + {10,20,30,40,50} = {11,22,33,44,55}

D:\code\CUDA_Test\CudaRuntime\x64\Debug\CudaRuntime.exe (process 14228) exited with code 0.
To automatically close the console when debugging stops, enable Tools->Options->Debugging->Automatically close the console when debugging stops.
Press any key to close this window . . .

```

Conclusion:
Congratulations! You have successfully set up the CUDA development environment and run a simple "Hello World" program. This is just the beginning of your journey with CUDA. As you continue to learn and explore CUDA, you will be able to leverage the power of GPUs for various computational tasks. Happy coding!