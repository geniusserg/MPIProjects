// Lab 1
// Danilov Sergey
// Matrix and vector mult


#define N 45000 // 45000x45000 matrix
#define M N
#define NUM_ATTEMPTS 15
#define PROC_MAX 6

#include <iostream>
#include "omp.h"
#include <math.h>
#include <vector>
#include <fstream>

long int* a = new long int[N*M]();
long int* b = new long int[N]();
long int* c = new long int[M]();

double block_parallel_for(int threads, long int* a, long int* b, long int* c) {
    double time_start = omp_get_wtime();
    omp_set_num_threads(threads);

    for (int i = 0; i < M; i++) {
        c[i] = 0;
    }

    #pragma omp parallel for
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            c[i] = c[i] + a[i*N+j]*b[j] ;
        }
    }

    return omp_get_wtime() - time_start;
}

double block_parallel_manual(int threads, long int* a, long int* b, long int* c) {
    double time_start = omp_get_wtime();

    int n_on_thread = M / threads;
    omp_set_num_threads(threads);

    for (int i = 0; i < M; i++) {
        c[i] = 0;
    }

    #pragma omp parallel 
    {
        int current_thread = omp_get_thread_num();
        for (int i = current_thread*n_on_thread; i < ((current_thread+1) * n_on_thread); i++) {
            for (int j = 0; j < N; j++) {
                c[i] = c[i] + a[i * N + j] * b[j];
            }
        }
    }

    // if some elements remain
    int start = threads * n_on_thread;
    for (int i = start; i < M; i++) {
        c[i] = 0;
        for (int j = 0; j < N; j++) {
            c[i] = c[i] + a[i * N + j] * b[j];
        }
    }

    return omp_get_wtime() - time_start;
}


void print_result(double* time_array, int value) {
    // print proc, time, accell, effic
    std::ofstream file_result;
    file_result.open("file.txt", std::ios::app);
    double min_time = time_array[value - 1];
    double max_time = time_array[0];
    double p = 1;
    for (int i = 0; i < value; i++) {
        p = pow(2, i);
        double time = time_array[i];
        double speedup = max_time / time;
        double efficency = speedup / p;
        std::cout << " " << p;
        file_result << " " << p;
        std::cout << "  " << time;
        file_result << "  " << time;
        std::cout << "  " << speedup;
        file_result << "  " << speedup;
        std::cout << "  " << efficency << "\n";
        file_result << "  " << efficency << "\n";
    }
    file_result << "\n\n";
    file_result.close();
}

void print_c_vector() {
    for (int i = 0; i < M; i++) 
        std::cout << c[i] << " ";
    std::cout << "\n\n";
}

void collect_and_print_statistic(double (*method)(int, long int*, long int*, long int*)) {
    double array[PROC_MAX];
    for (int i = 0; i < PROC_MAX; i++) {
        array[i] = 0;
    }

    // collect several times and get average
    for (int attempt = 0; attempt < NUM_ATTEMPTS; attempt++) {
        for (int i = 0; i < PROC_MAX; i++) {
            array[i] += method(pow(2, i), a, b, c);
        }
    }
    for (int i = 0; i < PROC_MAX; i++) {
        array[i] = array[i] / NUM_ATTEMPTS;
    }

    print_result(array, PROC_MAX);
}

int main()
{
    //file for report
    std::ofstream file_result;
    file_result.open("file.txt");
    file_result.close();

    //define arrays
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            a[i*N+j] = 10000;
        }
    }
    for (int i = 0; i < M; i++) {
        c[i] = 0;
    }
    for (int i = 0; i < N; i++) {
        b[i] = 2222;
    }

    std::cout << "Manual threads\n";
    std::cout << " P  T    A     E" << "\n";
    collect_and_print_statistic(block_parallel_manual);

    std::cout << "Parallel for\n ";
    std::cout << " P  T    A     E" << "\n";
    collect_and_print_statistic(block_parallel_for);

    delete[] a;
    delete[] b;
    delete[] c;
}
