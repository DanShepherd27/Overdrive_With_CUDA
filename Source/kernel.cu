#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./kernel.cuh"
#include "./PreciseTimer.h"
#include <stdio.h>

__global__ void applyGainAndNormalizeKernel(float *samplesOfChannel, int gain, float originalMagnitude, int arrayLength) {
    // Get thread ID.
    int threadID = blockDim.x * blockIdx.x + threadIdx.x;
    bool endReached = arrayLength - threadID <= 0;

    /*if (threadID == 10000) {
        printf("ThreadID: %d \n", threadID);
        printf("Value before: %f \n", samplesOfChannel[threadID]);
    }*/
    
    if (!endReached) {
        // Apply gain
        samplesOfChannel[threadID] *= gain;

        // Normalize value
        if (samplesOfChannel[threadID] > originalMagnitude) {
            samplesOfChannel[threadID] = originalMagnitude;
        }
        else if (samplesOfChannel[threadID] < -1 * originalMagnitude) {
            samplesOfChannel[threadID] = -1 * originalMagnitude;
        }
    }

    /*if (threadID == 10000) {
        printf("Value after: %f \n", samplesOfChannel[threadID]);
    }*/
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
    /*const float* samplesOfLeftChannel = samples_by_channels[0];
    printf("[BEFORE] Value of the 10.000th sample: %f \n", samplesOfLeftChannel[10000]);*/
    printf("Kernel started...\n");
    CPreciseTimer timer;
    timer.StartTimer();
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
        applyGainAndNormalizeKernel << < gridSize, blockSize >> > (dev_samplesOfChannel, gain, originalMagnitude, arrayLength);
        
        // Copy result back to host memory.
        cudaMemcpy((void*)samples_by_channels[channelNumber], dev_samplesOfChannel, arraySize, cudaMemcpyDeviceToHost);
    
    }
    timer.StopTimer();
    printf("Timer: %f ms.\n", timer.GetTimeMilliSec());

    /*printf("CUDA has run. \n");
    printf("[AFTER] Value of the 10.000th sample: %f \n", samplesOfLeftChannel[10000]);*/
}