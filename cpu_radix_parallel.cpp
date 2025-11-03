#ifndef CPU_PARALLEL_RADIX_SORT_H
#define CPU_PARALLEL_RADIX_SORT_H

#include "radix_sort.h"
#include <cstdlib>
#include <iostream>
#include <thread>
#include <vector>

class CPURadixSortParallel : public RadixSort {
private:
    void CountingSort(long int exp, long int* output, int thread_id, int num_threads) {
        int countArray[10] = { 0 };
        
        for (int i = thread_id; i < n; i += num_threads) {
            countArray[(table[i] / exp) % 10]++;
        }

        for (int i = 1; i < 10; i++)
            countArray[i] += countArray[i - 1];

        for (int i = n - 1; i >= 0; i--) {
            int index = (table[i] / exp) % 10;
            if (i % num_threads == thread_id) {
                output[countArray[index] - 1] = table[i];
                countArray[index]--;
            }
        }
    }

public:
    CPURadixSortParallel(int size, long int* data) : RadixSort(size, data) {}

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
            int num_threads = std::thread::hardware_concurrency();

            for (int i = 0; i < num_threads; i++) {
                threads.emplace_back(&CPURadixSortParallel::CountingSort, this, exp, sortedArray, i, num_threads);
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
