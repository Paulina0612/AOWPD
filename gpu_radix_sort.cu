#include "radix_sort.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define RADIX 16
#define BITS_PER_PASS 4

// Count kernel
__global__ void count_kernel(long int *d_input, int *d_block_digit_counts, int n, int shift)
{
    __shared__ int local_counts[RADIX];
    __shared__ int warp_counts[8][RADIX];  // 256 threads / 32 = 8 warps

    int lid = threadIdx.x;
    int blockId = blockIdx.x;
    int gid = blockId * blockDim.x + lid;
    int warp_id = lid / 32;
    int lane = lid % 32;

    // Initialize shared counts
    if (lid < RADIX)
    {
        local_counts[lid] = 0;
    }
    // Initialize warp counts
    if (lid < 8 * RADIX)
    {
        warp_counts[lid / RADIX][lid % RADIX] = 0;
    }
    __syncthreads();

    // Each thread reads one element
    int digit = -1;
    if (gid < n)
    {
        unsigned long int uvalue = (unsigned long int)d_input[gid];
        digit = (uvalue >> shift) & (RADIX - 1);
    }

    // Warp-level counting using ballot and popc
    #pragma unroll
    for (int d = 0; d < RADIX; d++)
    {
        unsigned int mask = __ballot_sync(0xffffffff, digit == d);
        if (lane == 0)
        {
            warp_counts[warp_id][d] = __popc(mask);
        }
    }
    __syncthreads();

    // Reduce warp counts to block counts
    if (lid < RADIX)
    {
        int sum = 0;
        #pragma unroll
        for (int w = 0; w < 8; w++)
        {
            sum += warp_counts[w][lid];
        }
        local_counts[lid] = sum;
    }
    __syncthreads();

    // Write block's counts to global memory
    if (lid < RADIX)
    {
        d_block_digit_counts[blockId * RADIX + lid] = local_counts[lid];
    }
}

// Scatter kernel using parallel scan within shared memory
__global__ void write_kernel(long int *d_input, long int *d_output, int *d_block_offsets, int n, int shift)
{
    __shared__ int digit_offsets[RADIX];
    __shared__ int scan_temp[256];  // For prefix scan, matches block size
    
    int lid = threadIdx.x;
    int blockId = blockIdx.x;
    int gid = blockIdx.x * blockDim.x + lid;

    // Load block offsets into shared memory
    if (lid < RADIX)
    {
        digit_offsets[lid] = d_block_offsets[blockId * RADIX + lid];
    }
    __syncthreads();

    // Read element and get its digit
    long int value = 0;
    int digit = -1;
    
    if (gid < n)
    {
        value = d_input[gid];
        unsigned long int uvalue = (unsigned long int)value;
        digit = (uvalue >> shift) & (RADIX - 1);
    }

    // Process each digit value, compute positions for all threads with that digit
    for (int d = 0; d < RADIX; d++)
    {
        // Each thread marks 1 if it has this digit, otherwise 0
        int has_digit = (digit == d) ? 1 : 0;
        scan_temp[lid] = has_digit;
        __syncthreads();
        
        // Parallel prefix sum
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int val = 0;
            if (lid >= stride)
            {
                val = scan_temp[lid - stride];
            }
            __syncthreads();
            scan_temp[lid] += val;
            __syncthreads();
        }
        
        // Write element if this thread has this digit
        if (gid < n && digit == d)
        {
            int local_pos = scan_temp[lid] - 1;
            int output_pos = digit_offsets[d] + local_pos;
            d_output[output_pos] = value;
        }
        __syncthreads();
    }
}

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

        // Block and grid dimensions
        const int threadsPerBlock = 256;
        const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // Allocate device memory for input/output buffers
        long int *d_input, *d_output;
        cudaMalloc(&d_input, n * sizeof(long int));
        cudaMalloc(&d_output, n * sizeof(long int));

        // Allocate device memory for counting and offset arrays
        int *d_block_counts, *d_block_offsets;
        cudaMalloc(&d_block_counts, blocksPerGrid * RADIX * sizeof(int));
        cudaMalloc(&d_block_offsets, blocksPerGrid * RADIX * sizeof(int));

        // Copy input data to device
        cudaMemcpy(d_input, table, n * sizeof(long int), cudaMemcpyHostToDevice);

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
            
            // Count per-block per-digit occurrences
            count_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_current_input, d_block_counts, n, shift);
            cudaDeviceSynchronize();
            
            // Compute prefix sum to determine output positions
            int *h_block_counts = new int[blocksPerGrid * RADIX];
            int *h_block_offsets = new int[blocksPerGrid * RADIX];
            int *h_global_digit_counts = new int[RADIX];
            int *h_global_digit_offsets = new int[RADIX];

            cudaMemcpy(h_block_counts, d_block_counts, blocksPerGrid * RADIX * sizeof(int), cudaMemcpyDeviceToHost);

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

            cudaMemcpy(d_block_offsets, h_block_offsets, blocksPerGrid * RADIX * sizeof(int), cudaMemcpyHostToDevice);
            
            delete[] h_block_counts;
            delete[] h_block_offsets;
            delete[] h_global_digit_counts;
            delete[] h_global_digit_offsets;

            // Scatter elements to output positions
            write_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_current_input, d_current_output, d_block_offsets, n, shift);
            cudaDeviceSynchronize();

            // Swap buffers for next pass
            long int *temp = d_current_input;
            d_current_input = d_current_output;
            d_current_output = temp;
        }

        // Copy sorted result back to host
        cudaMemcpy(table, d_current_input, n * sizeof(long int), cudaMemcpyDeviceToHost);

        // Clean up device memory
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_block_counts);
        cudaFree(d_block_offsets);
    }
};