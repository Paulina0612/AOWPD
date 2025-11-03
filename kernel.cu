#include <iostream>
#include <chrono>
#include <string>
#include "cpu_radix_sort.cpp"
#include "cpu_radix_parallel.cpp"
#include "gpu_radix_sort.cu"

using namespace std;

// helper function to print table (for debugging purposes)
void PrintTable(long int* table, int size)
{
    for (int i = 0; i < size; i++)
    {
        cout << table[i] << " ";
    }
    cout << endl;
}

int main()
{
    string sort_names[3];
    sort_names[0] = "CPU Sequential Radix Sort";
    sort_names[1] = "CPU Parallel Radix Sort";
    sort_names[2] = "GPU Parallel Radix Sort";

    // Declaring variables
    bool program = 1;
    int sample;
    int n;

    float sort_times[3];
    chrono::steady_clock::time_point start;
    chrono::steady_clock::time_point end;

    srand(time(NULL));

    while (program)
    {
        // Cleaning sort times
        for (int i = 0; i < 3; i++)
        {
            sort_times[i] = 0;
        }

        // Setting sample size
        cout << "Set sample size: ";
        cin >> sample;

        // Creating tables
        cout << "Set number of table elements: ";
        cin >> n;
        long int* tab = new long int[n];
        long int* tab_copy = new long int[n];

        // Initialize table with sequential indexes
        for (long int i = 0; i < n; i++)
        {
            tab[i] = i;
            tab_copy[i] = tab[i];
        }

        // Experiment
        for (int s = 0; s < sample; s++)
        {
            // Creating new random table for next sample by shuffling indexes
            for (long int i = 0; i < n; i++)
            {
                int los = rand() % n;
                swap(tab[i], tab[los]);
            }
            for (long int i = 0; i < n; i++)
            {
                tab_copy[i] = tab[i];
            }

            // Sorting and time counting
            for (int sort = 0; sort < 3; sort++)
            {
                switch (sort)
                {
                case 0:
                    {
                        CPUSequentialRadixSort sorter(n, tab);
                        start = chrono::steady_clock::now();
                        sorter.Sort();
                        end = chrono::steady_clock::now();
                    }
                    break;
                case 1:
                    {
                        CPURadixSortParallel sorter(n, tab);
                        start = chrono::steady_clock::now();
                        sorter.Sort();
                        end = chrono::steady_clock::now();
                    }
                    break;
                case 2:
                    {
                        GPUParallelRadixSort sorter(n, tab);
                        start = chrono::steady_clock::now();
                        sorter.Sort();
                        end = chrono::steady_clock::now();
                    }
                    break;
                }
                sort_times[sort] += chrono::duration_cast<chrono::microseconds>(end - start).count();

                // Checking if sorting was correctly done
                bool successful = true;
                for (int i = 1; i < n; i++) {
                    if (tab[i - 1] > tab[i]) {
                        successful = false;
                        break;
                    }
                }

                if (!successful)
                {
                    cout << "Sorting was not done correctly for " << sort_names[sort] << "!" << endl;
                    //PrintTable(tab, n);
                }

                // Bring back data
                for (long int i = 0; i < n; i++)
                {
                    tab[i] = tab_copy[i];
                }
            }
            cout << "End of sample: " << s + 1 << endl;
        }
        
        // Print mean time of execution
        cout << endl << "Results: " << endl;
        for (int i = 0; i < 3; i++)
        {
            cout << sort_names[i] << ": " << sort_times[i] / sample << " microseconds." << endl;
        }
        cout << endl;

        cout << "Do you want to run another test? (1 for yes, 0 for no): ";
        cin >> program;

        delete[] tab;
        delete[] tab_copy;
    }
    
    return 0;
}
