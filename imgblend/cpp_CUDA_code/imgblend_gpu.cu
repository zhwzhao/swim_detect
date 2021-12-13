#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "imgblend_gpu.h"
#include "cuda_utils.h"


// CUDA使用__global__来定义kernel
// __global__ void ball_query_kernel_cuda(int b, int n, int m, float radius, int nsample,

__global__ void imgblend_kernel_cuda(int width, int height, int left, int right, const int *__restrict__ new_xyz, const int *__restrict__ xyz, int *__restrict__ idx) {
    // threadIdx是一个三维的向量，可以用.x .y .z分别调用其三个维度。此处我们只初始化了第一个维度为THREADS_PER_BLOCK
    // blockIdx也是三维向量。我们初始化用的DIVUP(height, THREADS_PER_BLOCK), width分别对应blockIdx.x和blockIdx.y
    // blockDim代表block的长度
    if (blockIdx.y >= width || pt_idx >= left) return;

    int x = 0;

    int pt_idx;
    pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 针对指针数据，利用+的操作来确定数组首地址，相当于取new_xyz[bi,ni]
    int offset = blockIdx.y * left * 3 + pt_idx * 3;
    new_xyz += offset;
    xyz += offset;
    idx += offset;

    int src1pix = new_xyz[0] + new_xyz[1] + new_xyz[2];
    int src2pix = xyz[0] + xyz[1] + xyz[2];

    if(src2pix==0){
        idx[0] = new_xyz[0];
        idx[1] = new_xyz[1];
        idx[2] = new_xyz[2];
    }else if(src1pix==0){
        idx[0] = xyz[0];
        idx[1] = xyz[1];
        idx[2] = xyz[2];
    }else{
        float srclen = abs(pt_idx - left);
        float warplen = abs(pt_idx - right);
        float d = srclen/(srclen + warplen);

        idx[0] = int(new_xyz[0] * (1-d) + xyz[0] * d);
        idx[1] = int(new_xyz[1] * (1-d) + xyz[1] * d);
        idx[2] = int(new_xyz[2] * (1-d) + xyz[2] * d);
    }

}


// void imgblend_kernel_launcher_cuda(int b, int n, int m, float radius, int nsample

void imgblend_kernel_launcher_cuda(int width, int height, int left, int right, \
    const int *new_xyz, const int *xyz, int *idx) {

    // cudaError_t变量用来记录CUDA的err信息，在最后需要check
    cudaError_t err;
    // divup定义在cuda_utils.h,DIVUP(m, t)相当于把m个点平均划分给t个block中的线程，每个block可以处理THREADS_PER_BLOCK个线程。
    // THREADS_PER_BLOCK=256，假设我有m=1024个点，那就是我需要4个block，一共256*4个线程去处理这1024个点。
    // blockIdx.x(col), blockIdx.y(row)
    dim3 blocks(DIVUP(height, THREADS_PER_BLOCK), width);
    dim3 threads(THREADS_PER_BLOCK);

    // 可函数需要用<<<blocks, threads>>> 去指定调用的块数和线程数，总共调用的线程数为blocks1*threads
    imgblend_kernel_cuda<<<blocks, threads>>>(width, height, left, right, new_xyz, xyz, idx);

    // 如果cuda操作错误，则打印错误信息
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}
