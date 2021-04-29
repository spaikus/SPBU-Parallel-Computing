#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define is_master(id) !id

static int id;
static int num_threads;

void send_mes(int to)
{
    const int buffsiz = 15;
    char mes[buffsiz] = "Hello, world _";
    mes[buffsiz - 2] = '0' + to;

    MPI_Send(mes, buffsiz, MPI_CHAR, to, to, MPI_COMM_WORLD);

    printf("process[%d/%d]: sending message to process %d\n"
           "> %s\n\n",
           id, num_threads, to, mes);
}

void recv_mes(int from)
{
    MPI_Status status;
    MPI_Probe(from, id, MPI_COMM_WORLD, &status);

    int buffsiz;
    MPI_Get_count(&status, MPI_CHAR, &buffsiz);

    char *mes = malloc(buffsiz * sizeof(char));
    MPI_Recv(mes, buffsiz, MPI_CHAR, from, id, MPI_COMM_WORLD, &status);

    printf("process[%d/%d]: received message from process %d\n"
           "> %s\n\n",
           id, num_threads, from, mes);
}

void tran_mes(int from, int to)
{
    MPI_Status status;
    MPI_Probe(from, id, MPI_COMM_WORLD, &status);

    int buffsiz;
    MPI_Get_count(&status, MPI_CHAR, &buffsiz);

    char *mes = malloc(buffsiz * sizeof(char));
    MPI_Recv(mes, buffsiz, MPI_CHAR, from, id, MPI_COMM_WORLD, &status);

    printf("process[%d/%d]: received message from process %d\n"
           "> %s\n"
           "process[%d/%d]: sending message to process %d\n\n",
           id, num_threads, from, mes, id, num_threads, to);

    mes[buffsiz - 2] = '0' + to;
    MPI_Send(mes, buffsiz, MPI_CHAR, to, to, MPI_COMM_WORLD);
    free(mes);
}

void roundabout(void)
{
    if (is_master(id)) {
        printf("starting ring message exchange...\n\n");
    }

    if (num_threads == 1)
    {
        printf("process[%d/%d]: I am here alone, no process to send a message\n",
               id, num_threads);
    }
    else if (is_master(id))
    {
        send_mes(1);
        recv_mes(num_threads - 1);
    }
    else
    {
        tran_mes(id - 1, id + 1 == num_threads ? 0 : id + 1);
    }
    

    MPI_Barrier(MPI_COMM_WORLD);
    if (is_master(id)) {
        printf("ring message exchange is over\n\n");
    }
}

void master_slave(void)
{
    if (is_master(id)) {
        printf("starting master-slave message exchange...\n\n");
    }

    if (num_threads == 1)
    {
        printf("process[%d/%d]: I am here alone, no process to send a message\n",
               id, num_threads);
    }
    else if (is_master(id))
    {
        for (int process = 1; process < num_threads; ++process) {
            send_mes(process);
            recv_mes(process);
        }
    }
    else
    {
        tran_mes(0, 0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (is_master(id)) {
        printf("master-slave message exchange is over\n\n");
    }
}

void each_to_each(void)
{
    if (is_master(id)) {
        printf("starting everyone to everyone message exchange...\n\n");
    }

    if (num_threads == 1)
    {
        printf("process[%d/%d]: I am here alone, no process to send a message\n",
               id, num_threads);
    }
    else
    {
        for (int process = 0; process < num_threads; ++process)
        {
            if (id != process) {
                send_mes(process);
                recv_mes(process);
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (is_master(id)) {
        printf("everyone to everyone message exchange is over\n\n");
    }
}


int main(int argc, const char * argv[])
{
    /*
     MPI task1.2
     send messages in different ways:
     - roundabout
     - master-slave
     - everyone to everyone
     */

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);

    roundabout();
    master_slave();
    each_to_each();

    MPI_Finalize();
    return 0;
}

