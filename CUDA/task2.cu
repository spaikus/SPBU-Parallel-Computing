/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation. 
 * Any use, reproduction, disclosure, or distribution of this software 
 * and related documentation without an express license agreement from
 * NVIDIA Corporation is strictly prohibited.
 *
 * Please refer to the applicable NVIDIA end user license agreement (EULA) 
 * associated with this source code for terms and conditions that govern 
 * your use of this NVIDIA software.
 * 
 */

#include "stdio.h"
#include <stdlib.h>
#include <sys/time.h>


void sumMatrixOnHost(float *MatA, float *MatB, float *MatC, const unsigned nx, const unsigned ny)
{
    float *ia = MatA;
    float *ib = MatB;
    float *ic = MatC;
    float *end = ic + nx * ny;
    
    while (ic != end)
    {
        *ic = *ia + *ib;
        ++ia; ++ib; ++ic;
    }
}

__global__ void sumMatrixOnGPU2D(float *MatA, float *MatB, float *MatC, const unsigned nx, const unsigned ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}

__global__ void sumMatrixOnGPU1D(float *MatA, float *MatB, float *MatC, const unsigned nx, const unsigned ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx)
    {
        unsigned end = nx * ny;
        unsigned idx = ix;
        while (idx < end)
        {
            MatC[idx] = MatA[idx] + MatB[idx];
            idx += nx;
        }
    }
}

__global__ void sumMatrixOnGPUMix(float *MatA, float *MatB, float *MatC, const unsigned nx, const unsigned ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny)
    {
        MatC[idx] = MatA[idx] + MatB[idx];
    }
}


double cpuSecond();
char are_equal(float *a, float *b, unsigned n);

int main()
{
    /*
     CUDA task2
     block/thread speed comparison
     */

    const unsigned nx = 1 << 14;
    const unsigned ny = 1 << 14;
    const unsigned N = nx * ny;

    float *MatA = (float *)malloc(N * sizeof(float));
    float *MatB = (float *)malloc(N * sizeof(float));
    float *MatC = (float *)malloc(N * sizeof(float));
    float *Res = (float *)malloc(N * sizeof(float));

    sumMatrixOnHost(MatA, MatB, Res, nx, ny);


    dim3 block;
    dim3 grid;
    double time;
    
    float *dev_MatA, *dev_MatB, *dev_MatC;
    cudaMalloc(&dev_MatA, N * sizeof(float));
    cudaMalloc(&dev_MatB, N * sizeof(float));
    cudaMalloc(&dev_MatC, N * sizeof(float));

    cudaMemcpy(dev_MatA, MatA, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_MatB, MatB, N * sizeof(float), cudaMemcpyHostToDevice);

    // 2D grid - 2D block

    block = {32, 16};
    grid = {(nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y};

    time = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(dev_MatA, dev_MatB, dev_MatC, nx, ny);
    cudaDeviceSynchronize();
    time = cpuSecond() - time;
    
    for (unsigned i = 0; i < N; ++i)
    {
        MatC[i] = -1;
    }
    cudaMemcpy(MatC, dev_MatC, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("(%d, %d) (%d %d): %f s - %s\n", grid.x, grid.y, block.x, block.y, 
           time, are_equal(MatC, Res, N) ? "correct":"BAD");

    // 1D grid - 1D block
    
    block = {128, 1};
    grid = {(nx + block.x - 1) / block.x, 1};

    time = cpuSecond();
    sumMatrixOnGPU1D<<<grid, block>>>(dev_MatA, dev_MatB, dev_MatC, nx, ny);
    cudaDeviceSynchronize();
    time = cpuSecond() - time;
    
    for (unsigned i = 0; i < N; ++i)
    {
        MatC[i] = -1;
    }
    cudaMemcpy(MatC, dev_MatC, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("(%d, %d) (%d %d): %f s - %s\n", grid.x, grid.y, block.x, block.y, 
           time, are_equal(MatC, Res, N) ? "correct":"BAD");

    // 2D grid - 1D block
    
    block = {256, 1};
    grid = {(nx + block.x - 1) / block.x, ny};

    time = cpuSecond();
    sumMatrixOnGPUMix<<<grid, block>>>(dev_MatA, dev_MatB, dev_MatC, nx, ny);
    cudaDeviceSynchronize();
    time = cpuSecond() - time;

    for (unsigned i = 0; i < N; ++i)
    {
        MatC[i] = -1;
    }
    cudaMemcpy(MatC, dev_MatC, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("(%d, %d) (%d %d): %f s - %s\n", grid.x, grid.y, block.x, block.y, 
           time, are_equal(MatC, Res, N) ? "correct":"BAD");

    // custom configurations

    block = {1, 1};
    grid = {(nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y};

    time = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(dev_MatA, dev_MatB, dev_MatC, nx, ny);
    cudaDeviceSynchronize();
    time = cpuSecond() - time;
    printf("(%d, %d) (%d %d): %f s\n", grid.x, grid.y, block.x, block.y, time);


    block = {128, 128};
    grid = {(nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y};

    time = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(dev_MatA, dev_MatB, dev_MatC, nx, ny);
    cudaDeviceSynchronize();
    time = cpuSecond() - time;
    printf("(%d, %d) (%d %d): %f s\n", grid.x, grid.y, block.x, block.y, time);


    block = {nx, ny};
    grid = {(nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y};

    time = cpuSecond();
    sumMatrixOnGPU2D<<<grid, block>>>(dev_MatA, dev_MatB, dev_MatC, nx, ny);
    cudaDeviceSynchronize();
    time = cpuSecond() - time;
    printf("(%d, %d) (%d %d): %f s\n", grid.x, grid.y, block.x, block.y, time);


    return 0;
}


double cpuSecond()
{
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

char are_equal(float *a, float *b, unsigned n)
{
    for (unsigned i = 0; i < n; ++i)
    {
        if (a[i] != b[i])
        {
            return 0;
        }
    }

    return 1;
}