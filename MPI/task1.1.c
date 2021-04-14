#include <stdio.h>
#include <mpi.h>

int main(int argc, const char * argv[])
{
    /*
     MPI task1.1
     print process info
     */

    MPI_Init(&argc, &argv);

    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);

    int num_threads;
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);

    char process_name[MPI_MAX_PROCESSOR_NAME];
    int process_name_len;
    MPI_Get_processor_name(process_name, &process_name_len);

    printf("thread [%d/%d], process %s\n", id, num_threads, process_name);

    MPI_Finalize();
    return 0;
}

