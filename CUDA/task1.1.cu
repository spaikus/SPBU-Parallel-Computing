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

#define N 10

__host__ void add(int *a, int *b, int *c)
{
    int id = 0;
    while (id < N)
    {
        c[id] = a[id] + b[id];
        id += 1;
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
     CUDA task1.1
     block/thread parallelize vector sum
     */

    int a[N], b[N], c[N];

    // fill the arrays 'a' and 'b' on the CPU
    for (int i = 0; i < N; i++)
    {
        a[i] = -i;
        b[i] = i * i;
    }

    //device memory allocation
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc(&dev_a, sizeof(int) * N);
    cudaMalloc(&dev_b, sizeof(int) * N);
    cudaMalloc(&dev_c, sizeof(int) * N);

    cudaMemcpy(dev_a, a, sizeof(int) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, sizeof(int) * N, cudaMemcpyHostToDevice);


    // display the host results
    add(a, b, c);

    printf("host:\n");
    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    for (int i = 0; i < N; i++)
    {
        c[i] = 0;
    }

    printf("\n");

    // display the device (threads) results
    printf("device (threads):\n");
    add_p<<<1, N>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }

    for (int i = 0; i < N; i++)
    {
        c[i] = 0;
        // dev_c[i] = 0;
    }

    printf("\n");

    // display the device (blocks) results
    printf("device (blocks):\n");
    add_p<<<N, 1>>>(dev_a, dev_b, dev_c);
    cudaMemcpy(c, dev_c, sizeof(int) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
    {
        printf("%d + %d = %d\n", a[i], b[i], c[i]);
    }


    //freeing device memory
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    return 0;
}