#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define is_even(a) !(a & 1)

static int id;
static int num_threads;

int * rand_int_arr(size_t len);
void mergesort_seq(int *arr, size_t len);
void mergesort_par_multimerge(int *arr, size_t len);
void mergesort_par_singlemerge(int *arr, size_t len);
char is_sorted(int *arr, size_t len);
char arrays_are_equal(int *arr1, int *arr2, size_t len);


int main(int argc, const char * argv[])
{
    /*
     MPI task2
     merge sort
     */

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);

    int num_tests = 10;
    int test_from = 5000, test_to = 25000;
    int test_step = 5000;

    if (!id)
    {
        const char *out_file_name = "task2_res.txt";

        FILE *output = fopen(out_file_name, "w");
        fprintf(output,
                "Время в мкс, количество потоков: %d\n"
                "| Размер | Последовательный | Параллельный (однопоточный) | Параллельный (многопоточный) | Ускорение (однопоточный) | Ускорение (многопоточный) |\n"
                "| --- | :---: | :---: | :---: | :---: | :---: |\n", num_threads);

        for (int test = test_from; test <= test_to; test += test_step)
        {
            double ts = .0, tsmp = .0, tmmp = .0;
            double time_point;

            for (int i = 0; i < num_tests; ++i)
            {
                int *arr = rand_int_arr(test);

                int *arr_mm = malloc(test * sizeof(int));
                memcpy(arr_mm, arr, test * sizeof(int));

                int *arr_sm = malloc(test * sizeof(int));
                memcpy(arr_sm, arr, test * sizeof(int));


                time_point = MPI_Wtime();
                mergesort_seq(arr, test);
                ts += MPI_Wtime() - time_point;

                time_point = MPI_Wtime();
                mergesort_par_singlemerge(arr_sm, test);
                tsmp += MPI_Wtime() - time_point;

                time_point = MPI_Wtime();
                mergesort_par_multimerge(arr_mm, test);
                tmmp += MPI_Wtime() - time_point;


                free(arr);
                free(arr_sm);
                free(arr_mm);
            }
            ts /= num_tests / 1e6;
            tsmp /= num_tests / 1e6;
            tmmp /= num_tests / 1e6;

            fprintf(output, "| %d | %.0f | %.0f | %.0f | %.2f | %.2f |\n",
                    test, ts, tsmp, tmmp, ts/tsmp, ts/tmmp);
        }

        fclose(output);
    } else {
        for (int test = test_from; test <= test_to; test += test_step)
        {
            for (int i = 0; i < num_tests; ++i)
            {
                mergesort_par_singlemerge(0, 0);
                mergesort_par_multimerge(0, 0);
            }
        }
    }

    MPI_Finalize();
    return 0;
}


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

void mergesort_(int *arr, int *cpy, size_t len, size_t blocklen)
{
    char is_swapped = 0;
    for (size_t doublelen = blocklen << 1; blocklen < len; blocklen = doublelen, doublelen <<= 1)
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

void mergesort_seq(int *arr, size_t len)
{
    int *cpy = malloc(len * sizeof(int));
    mergesort_(arr, cpy, len, 1);
    free(cpy);
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

void mergesort_par_prep(int **arr, size_t *len, int **cpy,
                        int *sendcounts, int *displs)
{
    int delta = (*len + num_threads - 1) / num_threads;

    sendcounts[0] = delta;
    displs[0] = 0;

    for (int i = 1; i < num_threads; ++i) {
        sendcounts[i] = delta;
        displs[i] = displs[i-1] + delta;
    }
    sendcounts[num_threads-1] = *len - delta * (num_threads-1);

    if (id) {
        *len = sendcounts[id];
        *arr = malloc(*len * num_sections() * sizeof(int));
        *cpy = malloc(*len * num_sections() * sizeof(int));
    } else {
        *cpy = malloc((*len) * sizeof(int));
        *len = *sendcounts;
    }

    MPI_Scatterv(*arr, sendcounts, displs,
                 MPI_INT, *arr, *sendcounts,
                 MPI_INT, 0, MPI_COMM_WORLD);
}

void mergesort_par_multimerge(int *arr, size_t len)
{
    MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if (len < num_threads) {
        if (!id) {
            mergesort_seq(arr, len);
        }
        return;
    }

    int *cpy;
    int *sendcounts = malloc(num_threads * sizeof(int));
    int *displs = malloc(num_threads * sizeof(int));
    int last_section = num_threads - 1;

    mergesort_par_prep(&arr, &len, &cpy, sendcounts, displs);

    char is_swapped = 0;

    mergesort_(arr, cpy, len, 1);

    int power2 = 1;
    int section = id;

    while (is_even(section) && section < last_section)
    {
        int from = id + power2;
        int fromfrom = from + power2;

        MPI_Status status;
        int meslen;
        if (num_threads <= fromfrom) {
            meslen = displs[num_threads - 1] + sendcounts[num_threads - 1] - displs[from];
        } else {
            meslen = displs[fromfrom] - displs[from];
        }

        MPI_Recv(arr + len, meslen, MPI_INT, from, id, MPI_COMM_WORLD, &status);

        merge(arr, cpy, len, len + meslen);
        pntswap(&arr, &cpy);
        is_swapped = !is_swapped;
        len += meslen;

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

    free(sendcounts);
    free(displs);
    free(cpy);
}

void mergesort_par_singlemerge(int *arr, size_t len)
{
    MPI_Bcast(&len, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    if (len < num_threads) {
        if (!id) {
            mergesort_seq(arr, len);
        }
        return;
    }

    int *cpy;
    int *sendcounts = malloc(num_threads * sizeof(int));
    int *displs = malloc(num_threads * sizeof(int));
    int last_section = num_threads - 1;

    mergesort_par_prep(&arr, &len, &cpy, sendcounts, displs);

    mergesort_(arr, cpy, len, 1);

    if (!id) {
        int from = num_threads - 1;
        MPI_Status status;
        MPI_Recv(arr + displs[from], sendcounts[from], MPI_INT, from, id, MPI_COMM_WORLD, &status);
    } else if (id == num_threads - 1) {
        MPI_Send(arr, sendcounts[id], MPI_INT, 0, 0, MPI_COMM_WORLD);
    }

    len = sendcounts[num_threads-1] + displs[num_threads-1];

    sendcounts[num_threads-1] = 0;
    if (id != num_threads - 1) {
        MPI_Gatherv(arr, *sendcounts, MPI_INT, arr, sendcounts, displs, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (!id) {
        mergesort_(arr, cpy, len, *sendcounts);
    }

    if (id) {
        free(arr);
    }

    free(sendcounts);
    free(displs);
    free(cpy);
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
