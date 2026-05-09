#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "mpi.h"

#define EPS 1e-12 // we hash defined this small value to make small values greater than zero to make the log function defined


// We have defined these variables as global since they are used across multiple functions
MPI_Status status; 
int myrank, size, M;


// initialized the buffers that will be used for communication and computation, the *_og buffers hold the original data to be sent in the next iteration, while the other buffers hold the received data to be computed on  
double *data_at_D1, *data_at_D2;
double *data_at_D1_og, *data_at_D2_og;

int isSendD1, isSendD2, isRecvD1, isRecvD2; // flags to indicate whether the current process will send or receive data for D1 and D2


// a modularized function to handle the sending of data
void Send_data(int d, double *send, double *recv, int tag)
{
	// variables to check if the current process can send or receive data
	int isSend = (myrank + d < size);
	int isRecv = (myrank - d >= 0);


	// we essentially extend the in-class idea of doing one set of communications first and then the other in order to remove the serialization of sends and receives.

	// the class example had two way nearest neighbour communication, to imitate the same we divide the processes into two groups based on their rank and the distance d, and we let one group send first while the other group receives, and then we reverse the order. This way we can have all sends and receives happening in parallel without waiting for each other, thus improving performance.
	int group = (myrank / d) % 2;

	if (group == 0)
	{
		if (isSend)
			MPI_Send(send, M, MPI_DOUBLE, myrank + d, tag, MPI_COMM_WORLD);
		if (isRecv)
			MPI_Recv(recv, M, MPI_DOUBLE, myrank - d, tag, MPI_COMM_WORLD, &status);
	}
	else
	{
		if (isRecv)
			MPI_Recv(recv, M, MPI_DOUBLE, myrank - d, tag, MPI_COMM_WORLD, &status);
		if (isSend)
			MPI_Send(send, M, MPI_DOUBLE, myrank + d, tag, MPI_COMM_WORLD);
	}
}


// a function to compute the data received from D1 and D2
void Compute_data()
{
	if (isRecvD1)
	{
		for (int i = 0; i < M; i++)
			data_at_D1[i] *= data_at_D1[i];
	}

	if (isRecvD2)
	{
		for (int i = 0; i < M; i++)
		{
			// To ensure that input to log function is positive we always take the MOD and add a small value EPS
			// This was not mentioned in the assignment but we had to do it because we were getting NaN values due to negative values being passed to the log function
			double val = fabs(data_at_D2[i]) + EPS;
			data_at_D2[i] = log(val);
		}
	}
}

// a modularized function to handle the receiving of data, it essentially does the same thing as the Send_data function but in reverse order
void Receive_data(int d, double *send, double *recv, int tag)
{
	int isSend = (myrank + d < size);
	int isRecv = (myrank - d >= 0);

	int group = (myrank / d) % 2;

	if (group == 0)
	{
		if (isRecv)
			MPI_Send(send, M, MPI_DOUBLE, myrank - d, tag, MPI_COMM_WORLD);
		if (isSend)
			MPI_Recv(recv, M, MPI_DOUBLE, myrank + d, tag, MPI_COMM_WORLD, &status);
	}
	else
	{
		if (isSend)
			MPI_Recv(recv, M, MPI_DOUBLE, myrank + d, tag, MPI_COMM_WORLD, &status);
		if (isRecv)
			MPI_Send(send, M, MPI_DOUBLE, myrank - d, tag, MPI_COMM_WORLD);
	}
}

double max(double a, double b) { return a > b ? a : b; } // returns max between two doubles 

int main(int argc, char *argv[])
{
	MPI_Init(&argc, &argv);

	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	// if any arguement is missing we simply finalize and exit
	if (argc < 6)
	{
		MPI_Finalize();
		return 0;
	}

	M = atoi(argv[1]); // size of the data 
	int D1 = atoi(argv[2]); // first distance
	int D2 = atoi(argv[3]); // second distance
	int T = atoi(argv[4]); // number of iterations
	int seed = atoi(argv[5]); // seed for random number generation

	// allocating memory for the data buffers
	data_at_D1 = (double *)malloc(M * sizeof(double));
	data_at_D2 = (double *)malloc(M * sizeof(double));
	data_at_D1_og = (double *)malloc(M * sizeof(double));
	data_at_D2_og = (double *)malloc(M * sizeof(double));

	srand(seed + myrank);
	for (int i = 0; i < M; i++)
		data_at_D1_og[i] = data_at_D2_og[i] =
			(double)rand() * (myrank + 1) / 10000.0;

	// updating the flags 
	isSendD1 = (myrank + D1 < size);
	isSendD2 = (myrank + D2 < size);
	isRecvD1 = (myrank - D1 >= 0);
	isRecvD2 = (myrank - D2 >= 0);

	double stime = MPI_Wtime();

	for (int i = 0; i < T; i++)
	{

		// Forward Data to destinations
		Send_data(D1, data_at_D1_og, data_at_D1, 10);
		Send_data(D2, data_at_D2_og, data_at_D2, 20);

		// Compute 
		Compute_data();

		// Return Data to sources
		Receive_data(D1, data_at_D1, data_at_D1_og, 30);
		Receive_data(D2, data_at_D2, data_at_D2_og, 40);

		//if(i==T-1) continue; // no updation for the last iteration
		// Updation according to the rules given in the assignment
		if (isSendD1)
		{
			for (int k = 0; k < M; k++)
			{
				data_at_D1_og[k] = (unsigned long long)data_at_D1_og[k] % 100000;
			}
		}

		if (isSendD2)
		{
			for (int k = 0; k < M; k++)
			{
				data_at_D2_og[k] = data_at_D2_og[k] * 100000.0;
			}
		}
	}

	//double local_time = MPI_Wtime() - stime;
	//double total_time;


	// we use MPI reduce to get the maximum time taken by any process so as to take into account the slowest process which is the overall bottleneck
	//MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	double maxD1 = -DBL_MAX, maxD2 = -DBL_MAX;

	// each process calculates the maximum value in its data buffers and then we use MPI to get the overall maximum across all processes, this is done separately for D1 and D2
	if (isSendD1)
		for (int k = 0; k < M; k++)
			maxD1 = max(maxD1, data_at_D1_og[k]);

	if (isSendD2)
		for (int k = 0; k < M; k++)
			maxD2 = max(maxD2, data_at_D2_og[k]);

	if (myrank != 0)
	{
		// all valid senders send their maximum values to the root process, we use different tags for D1 and D2 to avoid any confusion in the root process when receiving the values
		if (isSendD1)
			MPI_Send(&maxD1, 1, MPI_DOUBLE, 0, 67, MPI_COMM_WORLD);
		if (isSendD2)
			MPI_Send(&maxD2, 1, MPI_DOUBLE, 0, 69, MPI_COMM_WORLD);
	}
	else
	{
		// the root process receives the maximum values from all valid senders and updates the overall maximum values for D1 and D2 accordingly, we use the same max function defined earlier to get the maximum value between the received value and the current maximum
		for (int r = 1; r < size; r++)
		{
			double temp;
			if (r + D1 < size)
			{
				MPI_Recv(&temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 67, MPI_COMM_WORLD, &status);
				maxD1 = max(maxD1, temp);
			}
			if (r + D2 < size)
			{
				MPI_Recv(&temp, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 69, MPI_COMM_WORLD, &status);
				maxD2 = max(maxD2, temp);
			}
		}
	}
	double local_time = MPI_Wtime() - stime;
	double total_time;


	// we use MPI reduce to get the maximum time taken by any process so as to take into account the slowest process which is the overall bottleneck
	MPI_Reduce(&local_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if(myrank == 0){
		printf("%lf %lf %lf\n", maxD1, maxD2, total_time);
	}
	// freeing the allocated memory for the data buffers
	free(data_at_D1);
	free(data_at_D2);
	free(data_at_D1_og);
	free(data_at_D2_og);

	MPI_Finalize();
	return 0;
}
