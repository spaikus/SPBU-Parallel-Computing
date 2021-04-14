#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define is_even(a) !(a & 1)

static int id;
static int num_threads;

void pntswap(void **a, void **b)
{
    void *tmp = *a;
    *a = *b;
    *b = tmp;
}

int * rand_int_arr(size_t len)
{
    int *arr = malloc(len * sizeof(int));
    for (size_t i = 0; i < len; ++i) {
        arr[i] = rand() % 100;
    }

    return arr;
}

void print_array(const int *arr, size_t len)
{
    const int *end = arr + len;
    while (arr != end)
    {
        printf("%d ", *arr);
        ++arr;
    }
    printf("\n");
}

void merge(const int *arr, int *cpy, size_t le, size_t re)
{
    size_t i = 0;
    size_t l = 0, r = le;

    while (l < le && r < re)
    {
        if (arr[r] < arr[l]) {
            cpy[i] = arr[r];
            ++i; ++r;
        } else {
            cpy[i] = arr[l];
            ++i; ++l;
        }
    }

    while (l < le) {
        cpy[i] = arr[l];
        ++i; ++l;
    }
    while (r < re) {
        cpy[i] = arr[r];
        ++i; ++r;
    }
}

void mergesort_seq(int *arr, size_t len)
{
    int *cpy = malloc(len * sizeof(int));
    char is_swapped = 0;
    for (size_t blocklen = 1, doublelen = 2; blocklen < len; blocklen = doublelen, doublelen <<= 1)
    {
        for (size_t i = 0; i < len; i += doublelen)
        {
            size_t le = blocklen, re = doublelen;
            if (len < i + le) {
                re = le = len - i;
            }
            if (len < i + re) {
                re = len - i;
            }

            merge(arr + i, cpy + i, le, re);
        }

        pntswap(&arr, &cpy);
        is_swapped = !is_swapped;
    }

    if (is_swapped)
    {
        for (size_t i = 0; i < len; ++i) {
            cpy[i] = arr[i];
        }
    }
}


int num_sections()
{
    int num = 1;
    int l = id, r = num_threads - 1;

    while (is_even(l) && l < r) {
        num <<= 1;
        l >>= 1; r >>= 1;
    }

    return num;
}

void mergesort_par(int *arr, size_t len)
{
    MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (!len) {
        return;
    }

    int *cpy;
    char is_swapped = 0;
    int *sendcounts = malloc(num_threads * sizeof(int));
    int last_section;

    {
        int *displs = malloc(num_threads * sizeof(int));
        int delta = (len + num_threads - 1) / num_threads;
        int i = 0;

        sendcounts[i] = delta;
        displs[i] = 0;
        ++i; len -= delta;

        while (delta < len) {
            sendcounts[i] = delta;
            displs[i] = displs[i-1] + delta;
            ++i; len -= delta;
        }
        if (len) {
            sendcounts[i] = len;
            displs[i] = displs[i-1] + delta;
            ++i;
        }
        last_section = i - 1;

        if (id && id < i)
        {
            len = sendcounts[id];
            arr = malloc(len * num_sections() * sizeof(int));
            if (is_even(id)) {
                cpy = malloc(len * num_sections() * sizeof(int));
            }
        }
        else if(id)
        {
            len = 0;
            arr = malloc(sizeof(int));
        }
        else
        {
            cpy = malloc((displs[i-1] + len) * sizeof(int));
            len = *sendcounts;
        }

        while (i < num_threads) {
            sendcounts[i] = 1;
            displs[i] = 0;
            ++i;
        }

        MPI_Scatterv(arr, sendcounts, displs,
                     MPI_INT, arr, *sendcounts,
                     MPI_INT, 0, MPI_COMM_WORLD);

        free(displs);
    }

    if (len)
    {
        mergesort_seq(arr, len);

        int power2 = 1;
        int section = id;

        while (is_even(section) && section < last_section)
        {
            int from = id + power2;

            if (sendcounts[from])
            {
                MPI_Status status;
                MPI_Probe(from, id, MPI_COMM_WORLD, &status);

                int meslen;
                MPI_Get_count(&status, MPI_INT, &meslen);
                MPI_Recv(arr + len, meslen, MPI_INT, from, id, MPI_COMM_WORLD, &status);

                merge(arr, cpy, len, len + meslen);
                pntswap(&arr, &cpy);
                is_swapped = !is_swapped;
                len += meslen;
            }

            power2 <<= 1;
            section >>= 1;
            last_section >>= 1;
        }

        if (id)
        {
            while (is_even(section)) {
                power2 <<= 1;
                section >>= 1;
            }

            int to = id - power2;
            MPI_Send(arr, len, MPI_INT, to, to, MPI_COMM_WORLD);
        }

    }


    free(sendcounts);
    if (id) {
        free(arr);
    }
    else
    {
        if (is_swapped)
        {
            pntswap(&arr, &cpy);
            for (size_t i = 0; i < len; ++i) {
                arr[i] = cpy[i];
            }
        }
    }

    if (is_even(id) && len) {
        free(cpy);
    }
}

char is_sorted(int *arr, size_t len)
{
    --len;
    for (size_t i = 0; i < len; ++i)
    {
        if (arr[i+1] < arr[i]) {
            return 0;
        }
    }

    return 1;
}

char arrays_are_equal(int *arr1, int *arr2, size_t len)
{
    while (len)
    {
        --len;
        if (arr1[len] != arr2[len]) {
            return 0;
        }
    }

    return 1;
}

int main(int argc, const char * argv[])
{
    /*
     MPI task2
     merge sort
     */

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);

    if (!id)
    {
        size_t len = 150000;
        int *arr = rand_int_arr(len);

        int *arrp = malloc(len * sizeof(int));
        memcpy(arrp, arr, len * sizeof(int));

        double stime;
        stime = MPI_Wtime();
        mergesort_seq(arr, len);
        stime = MPI_Wtime() - stime;

        double ptime;
        ptime = MPI_Wtime();
        mergesort_par(arrp, len);
        ptime = MPI_Wtime() - ptime;


        printf("(s) %fs %s\n", stime, is_sorted(arr, len) ? "sorted" : "BAD!");
        printf("(p) %fs %s\n", ptime, is_sorted(arrp, len) ? "sorted" : "BAD!");
        printf("speed-up %f, %s\n", stime/ptime, arrays_are_equal(arr, arrp, len) ? "equal" : "DIF!");

        free(arr);
        free(arrp);
    } else {
        mergesort_par(0, 0);
    }

    MPI_Finalize();
    return 0;
}

