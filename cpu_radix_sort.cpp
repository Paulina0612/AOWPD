#ifndef CPU_RADIX_SORT_H
#define CPU_RADIX_SORT_H

#include "radix_sort.h"
#include <cstdlib>
#include <iostream>

class CPUSequentialRadixSort : public RadixSort {
private:
    // Counting sort function for a specific digit position
    long int* CountingSort(long int exp) {
        // Output array that will hold the sorted numbers
        long int *sortedArray;
        sortedArray = (long int*)malloc(n * sizeof(long int));

        if (sortedArray == NULL) {
            std::cerr << "Nie udalo sie zaalokac pamieci!" << std::endl;
            exit(1);
        }

        // Initialize count array
        int i, countArray[10] = { 0 };

        // Store count of occurrences in countArray[]
        for (i = 0; i < n; i++) {
            countArray[(table[i] / exp) % 10]++;
        }

        // Change countArray[i] so that it contains actual position of this digit in sortedArray[]
        for (i = 1; i < 10; i++)
            countArray[i] += countArray[i - 1];

        // Build the output array
        for (i = n - 1; i >= 0; i--) {
            sortedArray[countArray[(table[i] / exp) % 10] - 1] = table[i];
            countArray[(table[i] / exp) % 10]--;
        }

        return sortedArray;
    }

public:
    CPUSequentialRadixSort(int size, long int* data) : RadixSort(size, data) {}

    const char* GetName() const override {
        return "CPU Sequential Radix Sort";
    }

    void Sort() override {
        // Find the maximum number to know the number of digits
        long int max = GetMax();

        // Radix sort loop
        for (long int exp = 1; max / exp > 0; exp *= 10) {
            table = CountingSort(exp);
        }
    }
};

#endif // CPU_RADIX_SORT_H
