#include <stdio.h>
#include <cuda_runtime.h>

__global__ void cuda_kernel_func()
{
    printf("Hello World from GPU!\n");
}

int main()
{
    cudaDeviceProp deviceProp;
    int dev = 0;
    cudaSetDevice(dev);

    cudaGetDeviceProperties(&deviceProp, dev);

        // Calculate total number of threads
    int total_threads = deviceProp.multiProcessorCount * deviceProp.maxThreadsPerBlock;

    printf("Total number of cores (SM's - Streaming multiprocessors): %d \n", deviceProp.multiProcessorCount);
    printf("Max threads per block : %d \n", deviceProp.maxThreadsPerBlock);
    printf("Total number of threads: %d \n", total_threads);
    printf("\n");
    printf("Warp size : %d \n", deviceProp.warpSize);
    printf("Max Blocks per MultiProcessor : %d \n", deviceProp.maxBlocksPerMultiProcessor);
    printf("Max grid dimensions : %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
    printf("Max block dimensions : %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("----other---\n");
    printf("Device name: %s\n", deviceProp.name);
    printf("Memory Clock Rate (KHz): %d\n", deviceProp.memoryClockRate);
    printf("Memory Bus Width (bits): %d\n", deviceProp.memoryBusWidth);
    printf("Peak Memory Bandwidth (GB/s): %f\n", 2.0 * deviceProp.memoryClockRate * (deviceProp.memoryBusWidth / 8) / 1.0e6);
    printf("Device copy overlap: ");
    if (deviceProp.deviceOverlap)
        printf("Enabled\n");
    else
        printf("Disabled\n");
    printf("Kernel execution timeout : ");
    if (deviceProp.kernelExecTimeoutEnabled)
        printf("Enabled\n");
    else
        printf("Disabled\n");

    // specifies the execution configuration of the kernel launch.
    // first parameter 1 = number of block(s)
    // second parameter 1 = number of thread(s) per block.
    cuda_kernel_func<<<1, 1>>>();
    cudaDeviceSynchronize();

    return 0; // success
}

/*
__device__ functions: These can only be called from the device (i.e., from within a __global__ or __device__ function).
__global__ functions: These are kernel functions that can be called from the host, but are executed on the device.
__host__ functions: These can only be called from the host.
The functions cudaSetDevice and cudaGetDeviceProperties are host functions, so they cannot be called from within a __global__ function. That’s why you’re seeing these errors.
*/

/* cudaDeviceSynchronize()
 
 is called after the kernel launch.
 This function blocks the CPU until the device has completed all preceding requested tasks.
 In this case, it ensures that the printf inside the kernel is executed and its output is flushed before the program ends.
 This way, you should be able to see the output immediately after the kernel execution.
 Please note that using cudaDeviceSynchronize() can affect the performance of your program because it forces the CPU to wait for the GPU. Therefore, it should be used judiciously.
*/