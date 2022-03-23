#define N 16

#include <stdlib.h>
#include <iostream>
#include "mpi.h"
#include "math.h"

int size, rank, ndim, subvector_size;
int ndims[2] = { 0,0 };
int period[2] = { 0, 0 };
int coords[2];
MPI_Comm com, row_comm, col_comm;

double* vectorTemp;
double* vectorOriginal;

void print_vector(int s, double* vectort) {
	for (int i = 0; i < s; i++) {
		std::cout << vectort[i] << " ";
	}
	std::cout << std::endl;
}

int compare(const void* x1, const void* x2)   // функция сравнения элементов массива
{
	return ( *(double*)x1 - *(double*)x2);              // если результат вычитания равен 0, то числа равны, < 0: x1 < x2; > 0: x1 > x2
}

void prepare_block(int a, double* tVector, double* tVectorOrig) {
	for (int i = a*subvector_size; i < a * subvector_size + subvector_size; i++) {
		tVector[i-a * subvector_size] = tVectorOrig[i];
	}
}

void populate_block(int a, double* tVector, double* tVectorOrig) {
	for (int i = a * subvector_size; i < a * subvector_size + subvector_size; i++) {
		tVectorOrig[i] = tVector[i - a * subvector_size];
	}
}

int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double start = MPI_Wtime();
	subvector_size = N / size;


	//prepare data
	if (rank == 0) {
		vectorOriginal = new double[N * N];
		for (int i = 0; i < N; i++) {
			vectorOriginal[i] = i;
		}
	}

	vectorTemp = new double[subvector_size * subvector_size];
	for (int i = 0; i < subvector_size; i++) {
		vectorTemp[i] = 0;
	}
	MPI_Status mStatus;

	// v distribution for all processes
	if (rank == 0) {
		for (int i = 0; i < size; i++) {
			if (i == 0) continue;
			prepare_block(i,  vectorTemp, vectorOriginal);
			MPI_Send(vectorTemp, subvector_size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
		}
		prepare_block(0, vectorTemp, vectorOriginal);
	}

	// receive v
	if (rank != 0) {
		MPI_Recv(vectorTemp, subvector_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, &mStatus);
	}

	qsort(vectorTemp, subvector_size, sizeof(double), compare);

	// send v'
	if (rank != 0) {
		print_vector(subvector_size, vectorTemp);
		MPI_Send(vectorTemp, subvector_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
	}

	// gather and construct C matrix
	if (rank == 0) {
		populate_block(0, vectorTemp, vectorOriginal);
		for (int i = 0; i < size; i++) {
			if (i == 0) continue;
			MPI_Recv(vectorTemp, subvector_size, MPI_DOUBLE, i, 1, MPI_COMM_WORLD, &mStatus);
			populate_block(i, vectorTemp, vectorOriginal);
		}
		delete[] vectorOriginal;
		double end = MPI_Wtime();
		std::cout << end - start;
	}

	MPI_Finalize();
	return 0;
}

