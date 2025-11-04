#include "radix_sort.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// CUDA error checking macro
#define CUDA_CHECK(err) { gpuAssert((err), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPU Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// Configuration: 16 radix bins (4 bits per pass)
#define RADIX 16
#define BITS_PER_PASS 4

// Kernel to count digit occurrences using shared memory
__global__ void count_kernel(long int *d_input, int *d_counts, int n, int shift)
{
    __shared__ int local_counts[RADIX];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int blockSize = blockDim.x;

    // Initialize local counts
    for (int i = lid; i < RADIX; i += blockSize)
    {
        local_counts[i] = 0;
    }
    __syncthreads();

    // Count digits in this block's portion of the array
    if (tid < n)
    {
        unsigned long int uvalue = (unsigned long int)d_input[tid];
        int digit = (uvalue >> shift) & (RADIX - 1);
        atomicAdd(&local_counts[digit], 1);
    }
    __syncthreads();

    // Write block's counts to global memory
    for (int i = lid; i < RADIX; i += blockSize)
    {
        atomicAdd(&d_counts[i], local_counts[i]);
    }
}

// Two-phase scatter: Phase 1 - compute per-block per-digit counts
__global__ void scatter_count_kernel(long int *d_input, int *d_block_digit_counts, int n, int shift, int num_blocks)
{
    __shared__ int local_counts[RADIX];

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lid = threadIdx.x;
    int blockSize = blockDim.x;
    int blockId = blockIdx.x;

    // Initialize local counts
    for (int i = lid; i < RADIX; i += blockSize)
    {
        local_counts[i] = 0;
    }
    __syncthreads();

    // Count digits in this block
    if (tid < n)
    {
        unsigned long int uvalue = (unsigned long int)d_input[tid];
        int digit = (uvalue >> shift) & (RADIX - 1);
        atomicAdd(&local_counts[digit], 1);
    }
    __syncthreads();

    // Write block's counts to global memory
    for (int i = lid; i < RADIX; i += blockSize)
    {
        d_block_digit_counts[blockId * RADIX + i] = local_counts[i];
    }
}

// Two-phase scatter: Phase 2 - scatter elements to output positions
__global__ void scatter_write_kernel(long int *d_input, long int *d_output, int *d_block_offsets, int n, int shift)
{
    extern __shared__ int shared_mem[];
    int *local_offsets = shared_mem;

    int lid = threadIdx.x;
    int blockId = blockIdx.x;
    int tid = blockIdx.x * blockDim.x + lid;

    // Initialize per-thread counter array
    for (int i = 0; i < RADIX; i++)
    {
        local_offsets[lid * RADIX + i] = 0;
    }
    __syncthreads();

    // Read element and mark its digit
    long int value = 0;
    int digit = -1;
    
    if (tid < n)
    {
        value = d_input[tid];
        unsigned long int uvalue = (unsigned long int)value;
        digit = (uvalue >> shift) & (RADIX - 1);
        local_offsets[lid * RADIX + digit] = 1;
    }
    __syncthreads();

    // Compute local position within block by counting elements with same digit before this thread
    int local_pos = 0;
    if (tid < n)
    {
        for (int t = 0; t < lid; t++)
        {
            local_pos += local_offsets[t * RADIX + digit];
        }
    }

    // Write element to output at computed global position
    if (tid < n)
    {
        int global_offset = d_block_offsets[blockId * RADIX + digit];
        int output_pos = global_offset + local_pos;
        d_output[output_pos] = value;
    }
}

// GPU Parallel Radix Sort class declaration and implementation
class GPUParallelRadixSort : public RadixSort
{
public:
    GPUParallelRadixSort(int size, long int *data) : RadixSort(size, data) {}

    const char *GetName() const override
    {
        return "GPU Parallel Radix Sort";
    }

    void Sort() override
    {
        if (n <= 0)
            return;

        // Configure block and grid dimensions
        const int threadsPerBlock = 64;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate device memory for input/output buffers
        long int *d_input, *d_output;
        CUDA_CHECK(cudaMalloc(&d_input, n * sizeof(long int)));
        CUDA_CHECK(cudaMalloc(&d_output, n * sizeof(long int)));

        // Allocate device memory for counting and offset arrays
        int *d_block_counts, *d_block_offsets;
        CUDA_CHECK(cudaMalloc(&d_block_counts, blocksPerGrid * RADIX * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_block_offsets, blocksPerGrid * RADIX * sizeof(int)));

        // Copy input data to device
        CUDA_CHECK(cudaMemcpy(d_input, table, n * sizeof(long int), cudaMemcpyHostToDevice));

        // Calculate number of passes needed based on maximum value
        long int maxVal = GetMax();
        unsigned long int umaxVal = (unsigned long int)maxVal;
        int num_bits = 0;
        while ((umaxVal >> num_bits) > 0)
            num_bits++;
        
        int num_passes = (num_bits + BITS_PER_PASS - 1) / BITS_PER_PASS;

        // Perform radix sort passes
        long int *d_current_input = d_input;
        long int *d_current_output = d_output;

        for (int pass = 0; pass < num_passes; pass++)
        {
            int shift = pass * BITS_PER_PASS;
            
            // Step 1: Count per-block per-digit occurrences
            scatter_count_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_current_input, d_block_counts, n, shift, blocksPerGrid);
            CUDA_CHECK(cudaDeviceSynchronize());
            
            // Step 2: Compute prefix sum to determine output positions
            int *h_block_counts = new int[blocksPerGrid * RADIX];
            int *h_block_offsets = new int[blocksPerGrid * RADIX];
            int *h_global_digit_counts = new int[RADIX];
            int *h_global_digit_offsets = new int[RADIX];
            
            CUDA_CHECK(cudaMemcpy(h_block_counts, d_block_counts, blocksPerGrid * RADIX * sizeof(int), cudaMemcpyDeviceToHost));

            // Sum counts for each digit across all blocks
            for (int digit = 0; digit < RADIX; digit++)
            {
                h_global_digit_counts[digit] = 0;
                for (int block = 0; block < blocksPerGrid; block++)
                    h_global_digit_counts[digit] += h_block_counts[block * RADIX + digit];
            }
            
            // Compute global starting position for each digit
            h_global_digit_offsets[0] = 0;
            for (int digit = 1; digit < RADIX; digit++)
                h_global_digit_offsets[digit] = h_global_digit_offsets[digit - 1] + h_global_digit_counts[digit - 1];
            
            // Compute starting position for each block's contribution to each digit
            for (int digit = 0; digit < RADIX; digit++)
            {
                int offset = h_global_digit_offsets[digit];
                for (int block = 0; block < blocksPerGrid; block++)
                {
                    h_block_offsets[block * RADIX + digit] = offset;
                    offset += h_block_counts[block * RADIX + digit];
                }
            }

            CUDA_CHECK(cudaMemcpy(d_block_offsets, h_block_offsets, blocksPerGrid * RADIX * sizeof(int), cudaMemcpyHostToDevice));
            
            delete[] h_block_counts;
            delete[] h_block_offsets;
            delete[] h_global_digit_counts;
            delete[] h_global_digit_offsets;

            // Step 3: Scatter elements to output positions
            int shared_mem_size = RADIX * threadsPerBlock * sizeof(int);
            scatter_write_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_current_input, d_current_output, d_block_offsets, n, shift);
            CUDA_CHECK(cudaDeviceSynchronize());

            // Swap buffers for next pass
            long int *temp = d_current_input;
            d_current_input = d_current_output;
            d_current_output = temp;
        }

        // Copy sorted result back to host
        CUDA_CHECK(cudaMemcpy(table, d_current_input, n * sizeof(long int), cudaMemcpyDeviceToHost));

        // Clean up device memory
        CUDA_CHECK(cudaFree(d_input));
        CUDA_CHECK(cudaFree(d_output));
        CUDA_CHECK(cudaFree(d_block_counts));
        CUDA_CHECK(cudaFree(d_block_offsets));
    }
};