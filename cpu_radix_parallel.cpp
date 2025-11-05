#ifndef CPU_PARALLEL_RADIX_SORT_H
#define CPU_PARALLEL_RADIX_SORT_H

#include "radix_sort.h"
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>
#include <barrier>
#include <functional>
#include <array>
#include <vector>

class CPURadixSortParallel : public RadixSort {
private:
    int num_threads;
    std::barrier<std::function<void()>> sync_point;
    std::vector<std::array<int, 10>> local_counts;
    std::array<int, 10> global_prefix;
    
    void CountingSort(long int exp, long int* output, int thread_id, int n_owned, int owned_idx) {
        local_counts[thread_id] = {0};
        
        for (int i = owned_idx; i < owned_idx + n_owned; i++) {
            local_counts[thread_id][(table[i] / exp) % 10]++;
        }

        sync_point.arrive_and_wait();

        if (thread_id == 0) {
            global_prefix = {0};
            for (int d = 0; d < 10; d++) {
                for (int t = 0; t < num_threads; t++) {
                    global_prefix[d] += local_counts[t][d];
                }
            }
            
            int sum = 0;
            for (int d = 0; d < 10; d++) {
                int count = global_prefix[d];
                global_prefix[d] = sum;
                sum += count;
            }
        }

        sync_point.arrive_and_wait();

        std::array<int, 10> thread_offset = global_prefix;
        for (int t = 0; t < thread_id; t++) {
            for (int d = 0; d < 10; d++) {
                thread_offset[d] += local_counts[t][d];
            }
        }

        for (int i = owned_idx; i < owned_idx + n_owned; i++) {
            int digit = (table[i] / exp) % 10;
            output[thread_offset[digit]++] = table[i];
        }

        sync_point.arrive_and_wait();
    }

public:
    CPURadixSortParallel(int size, long int* data) : RadixSort(size, data), 
        num_threads(std::thread::hardware_concurrency()), 
        sync_point(num_threads, [](){}),
        local_counts(num_threads) {}

    const char* GetName() const override {
        return "CPU Parallel Radix Sort";
    }

    void Sort() override {
        long int max = GetMax();
        long int* sortedArray = (long int*)malloc(n * sizeof(long int));

        if (sortedArray == NULL) {
            std::cerr << "Memory allocation failed!" << std::endl;
            exit(1);
        }

        for (long int exp = 1; max / exp > 0; exp *= 10) {
            std::vector<std::thread> threads;
            int start_idx = 0;
            for (int i = 0; i < num_threads; i++) {
                int n_owned = n / num_threads;
                if (i < (n % num_threads))
                    n_owned++;
                threads.emplace_back(&CPURadixSortParallel::CountingSort, this, exp, sortedArray, i, n_owned, start_idx);
                start_idx += n_owned;
            }

            for (auto& th : threads) {
                th.join();
            }

            std::copy(sortedArray, sortedArray + n, table);
        }

        free(sortedArray);
    }
};

#endif // CPU_PARALLEL_RADIX_SORT_H
