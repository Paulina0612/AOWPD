#include <iostream>
#include "cpu_radix_sort.cpp"
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

        double startTime = clock();

        switch (choice)
        {
        case 1: CPURadixSort(n, tablePointer); break;
        case 2:
            // TODO: Implement parallel CPU radix sort
            break;
        case 3:
            // TODO: Implement GPU radix sort
            break;
        case 4: exit(0); break;
        default:
            cout << "Nieprawidlowy wybor. Sprobuj ponownie." << endl;
            break;
        }

        double endTime = clock();
        if (choice >= 1 && choice <= 3) {
            double elapsedTime = double(endTime - startTime) / CLOCKS_PER_SEC;
            cout << "Czas wykonania: " << elapsedTime << " sekund." << endl;
        }
    }

}