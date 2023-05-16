#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./kernel.cuh"
#include <stdio.h>

__global__ void applyGainKernel(float *samplesOfChannel, int gain, int arrayLength) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    bool endReached = arrayLength - threadID <= 0;

    if (threadID == 10000) {
        printf("ThreadID: %d \n", threadID);
        printf("Value before: %f \n", samplesOfChannel[threadID]);
    }
    
    if (!endReached) {
        samplesOfChannel[threadID] *= gain;
    }

    if (threadID == 10000) {
        printf("Value after: %f \n", samplesOfChannel[threadID]);
    }
}

__global__ void normalizeSamplesKernel(float *samplesOfChannel, float originalMagnitude, int arrayLength) {
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    bool endReached = arrayLength - threadID <= 0;
    if (!endReached && samplesOfChannel[threadID] > originalMagnitude) {
        samplesOfChannel[threadID] = originalMagnitude;
    }
    if (!endReached && samplesOfChannel[threadID] < -1 * originalMagnitude) {
        samplesOfChannel[threadID] = -1 * originalMagnitude;
    }
}



/**
 * Wrapper function for the CUDA kernel function.
 * @param samples Pointer to an array of channel pointers that point to the arrays of samples.
 * @param numOfChannels Number of channels.
 * @param arrayLength Length of the array of samples per channel.
 * @param gain Gain to be applied to samples.
 * @param originalMagnitude The highest value of all samples before the effect is applied.
 */
void kernel(const float *const *samples_by_channels, int numOfChannels, int arrayLength, int gain, float originalMagnitude) {
    
    // Array size per channel.
    int arraySize = arrayLength * sizeof(float);
    const float* samplesOfLeftChannel = samples_by_channels[0];
    printf("[BEFORE] Value of the 10.000th sample: %f \n", samplesOfLeftChannel[10000]);

    for (size_t channelNumber = 0; channelNumber < numOfChannels; ++channelNumber)
    {
        // Initialize device pointers.
        float *dev_samplesOfChannel;

        // Allocate device memory.
        cudaMalloc((void**)&dev_samplesOfChannel, arraySize);

        // Transfer sample arrays to device.
        cudaMemcpy(dev_samplesOfChannel, samples_by_channels[channelNumber], arraySize, cudaMemcpyHostToDevice);
    

        // Calculate blocksize and gridsize.
        dim3 blockSize(1024);
        dim3 gridSize(arrayLength % blockSize.x == 0 ? arrayLength/blockSize.x : arrayLength / blockSize.x + 1);

        // Launch CUDA kernel.    
        applyGainKernel << < gridSize, blockSize >> > (dev_samplesOfChannel, gain, arrayLength);
        normalizeSamplesKernel << <gridSize, blockSize >> > (dev_samplesOfChannel, originalMagnitude, arrayLength);
        
        // Copy result back to host memory.
        cudaMemcpy((void*)samples_by_channels[channelNumber], dev_samplesOfChannel, arraySize, cudaMemcpyDeviceToHost);
    
    }

    printf("CUDA has run. \n");
    printf("[AFTER] Value of the 10.000th sample: %f \n", samplesOfLeftChannel[10000]);
}