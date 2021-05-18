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

#define BLOCK_DIM 4
#define GRID_DIM 5
#define THREADS BLOCK_DIM * GRID_DIM


__host__ void add(int *a, int *b, int *c)
{
    int tid = 0;    // this is CPU zero, so we start at zero
    while (tid < THREADS) {
        c[tid] = a[tid] + b[tid];
        tid += 1;   // we have one CPU, so we increment by one
    }
}

__global__ void add_p(int *a, int *b, int *c)
{
    printf("(%d %d)\n", threadIdx.x, blockIdx.x);
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    c[id] = a[id] + b[id];
}


int main( void ) 
{
    /*
     CUDA task1.2
     block-thread parallelize vector sum
     */

    int a[THREADS], b[THREADS], c[THREADS];

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < THREADS; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    //device memory allocation
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, sizeof(int) * THREADS);
    cudaMalloc(&dev_b, sizeof(int) * THREADS);
    cudaMalloc(&dev_c, sizeof(int) * THREADS);

    cudaMemcpy(dev_a, a, sizeof(int) * THREADS, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * THREADS, cudaMemcpyHostToDevice);


    // display the host results
    add(a, b, c);

    printf("host:\n");
    for (int i = 0; i < THREADS; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    for (int i = 0; i < THREADS; i++)
    {
        c[i] = 0;
    }

    printf("\n");

    // display the device results
    printf("device<<<%d, %d>>>:\n", GRID_DIM, BLOCK_DIM);
    add_p<<<GRID_DIM, BLOCK_DIM>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(c, dev_c, sizeof(int) * THREADS, cudaMemcpyDeviceToHost);
    for (int i = 0; i < THREADS; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }


    //freeing device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}