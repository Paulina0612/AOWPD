#include "radix_sort.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

// CUDA error check
#define CUDA_CHECK(err) { gpuAssert((err), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// CUDA kernel declarations
__global__ void histogram_kernel(long int* d_input, int* d_histogram, int n, long int exp);
__global__ void scatter_kernel(long int* d_input, long int* d_output, int* d_positions, int n, long int exp);

// GPU Parallel Radix Sort class declaration
class GPUParallelRadixSort : public RadixSort {
public:
    GPUParallelRadixSort(int size, long int* data);
    const char* GetName() const override;
    void Sort() override;
};

// CUDA kernel: compute histogram for a given digit (decimal LSD)
__global__ void histogram_kernel(long int* d_input, int* d_histogram, int n, long int exp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int digit = ((d_input[idx] / exp) % 10 + 10) % 10;
        atomicAdd(&d_histogram[digit], 1);
    }
}

// CUDA kernel: scatter elements to output based on positions
__global__ void scatter_kernel(long int* d_input, long int* d_output, int* d_positions, int n, long int exp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int digit = ((d_input[idx] / exp) % 10 + 10) % 10;
        int pos = atomicAdd(&d_positions[digit], 1);
        d_output[pos] = d_input[idx];
    }
}

// Konstruktor
GPUParallelRadixSort::GPUParallelRadixSort(int size, long int* data)
    : RadixSort(size, data) {
}

// Nazwa sortowania
const char* GPUParallelRadixSort::GetName() const
{
    return "GPU Parallel Radix Sort";
}

// G��wna funkcja sortowania
void GPUParallelRadixSort::Sort()
{
    if (n <= 0) return;

    long int maxVal = GetMax();

    // Allocate device memory
    long int* d_in_alloc;
    long int* d_out_alloc;
    CUDA_CHECK(cudaMalloc(&d_in_alloc, n * sizeof(long int)));
    CUDA_CHECK(cudaMalloc(&d_out_alloc, n * sizeof(long int)));

    long int* d_input = d_in_alloc;
    long int* d_output = d_out_alloc;

    int* d_histogram;
    int* d_positions;
    CUDA_CHECK(cudaMalloc(&d_histogram, 10 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_positions, 10 * sizeof(int)));

    // Copy input array to device
    CUDA_CHECK(cudaMemcpy(d_input, table, n * sizeof(long int), cudaMemcpyHostToDevice));

    // Host arrays for histogram and positions
    int h_histogram[10];
    int h_positions[10];

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    for (long int exp = 1; maxVal / exp > 0; exp *= 10)
    {
        CUDA_CHECK(cudaMemset(d_histogram, 0, 10 * sizeof(int)));

        histogram_kernel << <blocksPerGrid, threadsPerBlock >> > (d_input, d_histogram, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_histogram, d_histogram, 10 * sizeof(int), cudaMemcpyDeviceToHost));

        h_positions[0] = 0;
        for (int i = 1; i < 10; i++)
            h_positions[i] = h_positions[i - 1] + h_histogram[i - 1];

        CUDA_CHECK(cudaMemcpy(d_positions, h_positions, 10 * sizeof(int), cudaMemcpyHostToDevice));

        scatter_kernel << <blocksPerGrid, threadsPerBlock >> > (d_input, d_output, d_positions, n, exp);
        CUDA_CHECK(cudaDeviceSynchronize());

        // Swap input and output
        long int* temp = d_input;
        d_input = d_output;
        d_output = temp;
    }

    // Copy sorted array back to host
    CUDA_CHECK(cudaMemcpy(table, d_input, n * sizeof(long int), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_in_alloc));
    CUDA_CHECK(cudaFree(d_out_alloc));
    CUDA_CHECK(cudaFree(d_histogram));
    CUDA_CHECK(cudaFree(d_positions));
}
