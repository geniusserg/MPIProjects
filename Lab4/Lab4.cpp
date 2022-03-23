#define N 10000
#define PROPORTION 50

#include <stdlib.h>
#include <iostream>
#include "mpi.h"
#include "math.h"

int size, rank, subvector_size;

double* vResult;
double* vWork;

void print_vector(int s, double* vectort) {
	for (int i = 0; i < s; i++) {
		std::cout << vectort[i] << " ";
	}
	std::cout << std::endl;
}

int wasSortedForIter = 0;
int recvbuff = 0;

void sort(double* num, int size)
{
	for (int i = 0; i < size - 1; i++)
	{
		for (int j = (size - 1); j > i; j--) 
		{
			if (num[j - 1] > num[j]) 
			{
				int temp = num[j - 1]; 
				num[j - 1] = num[j];
				num[j] = temp;
				wasSortedForIter += 1;
			}
		}
	}
}


int main(int argc, char** argv) {
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	double start = MPI_Wtime();
	subvector_size = N / size;

	vWork = new double[2*subvector_size];
	vResult = new double[N];
	for (int i = 0; i < subvector_size; i++) {
		vWork[i] = rand()%100;
	}

	if (rank == 0) {
		for (int i = 0; i < N; i++) {
			vResult[i] = 0;
		}
	}
	MPI_Status mStatus;

	int p = subvector_size * ((double)PROPORTION / 100.0);

	sort(vWork, subvector_size);

	long long step = 0;
	while (true){
		wasSortedForIter = 0;
		if (rank % 2 == 0) {
			if (rank == size - 2) {
				MPI_Recv(vWork + subvector_size, subvector_size, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &mStatus);
				sort(vWork, subvector_size*2);
				MPI_Send(vWork + subvector_size, subvector_size, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
			}
			else {
				MPI_Recv(vWork + subvector_size, p, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &mStatus);
				sort(vWork, subvector_size+p);
				MPI_Send(vWork + subvector_size, p, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD);
			}
		}
		else{
			if (rank == size - 1) {
				MPI_Send(vWork, subvector_size, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
				MPI_Recv(vWork, subvector_size, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &mStatus);
			}
			else {
				MPI_Send(vWork, p, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD);
				MPI_Recv(vWork, p, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, &mStatus);
			}
		}

		if (rank % 2 != 0) {
			if (rank != size - 1) {
				MPI_Recv(vWork + subvector_size, p, MPI_DOUBLE, rank + 1, 2, MPI_COMM_WORLD, &mStatus);
				sort(vWork, subvector_size + p);
				MPI_Send(vWork + subvector_size, p, MPI_DOUBLE, rank + 1, 3, MPI_COMM_WORLD);
			}
		}
		else {
			if (rank != 0) {
				MPI_Send(vWork, p, MPI_DOUBLE, rank - 1, 2, MPI_COMM_WORLD);
				MPI_Recv(vWork, p, MPI_DOUBLE, rank - 1, 3, MPI_COMM_WORLD, &mStatus);
			}
		}
		
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Allreduce(&wasSortedForIter, &recvbuff, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
		if (recvbuff == 0) {
			break;
		}
		step += 1;
	}

	MPI_Gather(vWork, subvector_size, MPI_DOUBLE, vResult+(rank*subvector_size), subvector_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0) {
		//print_vector(N, vResult);
		
		double end = MPI_Wtime();
		std::cout << end - start;
	}

	MPI_Finalize();
	return 0;
}

