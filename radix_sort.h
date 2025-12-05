#ifndef RADIX_SORT_H
#define RADIX_SORT_H

#include <iostream>
const int BASE = 10;

class RadixSort {
protected:
    int n;
    long int* table;

    long int GetMax() {
        long int max = table[0];
        for (int i = 1; i < n; i++) {
            if (table[i] > max) {
                max = table[i];
            }
        }
        return max;
    }

public:
    RadixSort(int size, long int* data) : n(size), table(data) {}
    virtual ~RadixSort() {}

    virtual void Sort() = 0;
    virtual const char* GetName() const = 0;
};

#endif // RADIX_SORT_H
