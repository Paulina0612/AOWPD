#include <iostream>
#include <chrono>
#include "cpu_radix_sort.cpp"
#include "cpu_radix_parallel.cpp"
using namespace std;

long int *tablePointer;


void GenerateTable(int n) {
    srand(time(0));

    tablePointer = (long int*)malloc(n * sizeof(long int));
    if (tablePointer == NULL) {
        cerr << "Nie udalo sie zaalokac pamieci!" << endl;
        exit(1);
    }

    for (int i = 0; i < n; i++) {
        tablePointer[i] = rand();
    }
}


int main() {
    cout << "Podaj wielkosc tablicy do wygenerowania: " << endl;
    int n;
    cin >> n;

    GenerateTable(n);

    while (true){
        cout << "Wybierz opcje: " << endl;
        cout << "1. Sortowanie tablicy za pomoca sekwencyjnego sortowania pozycyjnego na CPU" << endl;
        cout << "2. Sortowanie tablicy za pomoca rownoleglego sortowania pozycyjnego na CPU." << endl;
        cout << "3. Sortowanie tablicy za pomoca sortowania pozycyjnego na GPU." << endl;
        cout << "4. Wyjscie z programu." << endl;

        int choice;
        cin >> choice;

        switch (choice)
        {
        case 1: 
            {
                CPUSequentialRadixSort sorter(n, tablePointer);
                cout << "Sortowanie za pomoca: " << sorter.GetName() << endl;
                auto startTime = chrono::high_resolution_clock::now();
                sorter.Sort();
                auto endTime = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> elapsedTime = endTime - startTime;
                cout << "Czas wykonania: " << elapsedTime.count() << " ms." << endl;
            }
            break;
        case 2:
            {
                CPURadixSortParallel sorter(n, tablePointer);
                auto startTime = chrono::high_resolution_clock::now();
                sorter.Sort();
                auto endTime = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> elapsedTime = endTime - startTime;
                cout << "Czas wykonania: " << elapsedTime.count() << " ms." << endl;
            }
            break;
        case 3:
            {
                auto startTime = chrono::high_resolution_clock::now();
                // TODO: Implement GPU radix sort
                auto endTime = chrono::high_resolution_clock::now();
                chrono::duration<double, milli> elapsedTime = endTime - startTime;
                cout << "Czas wykonania: " << elapsedTime.count() << " ms." << endl;
            }
            break;
        case 4: exit(0); break;
        default:
            cout << "Nieprawidlowy wybor. Sprobuj ponownie." << endl;
            break;
        }
    }

}
