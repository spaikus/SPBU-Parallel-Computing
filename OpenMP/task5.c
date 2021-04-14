#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static unsigned thread_num = 8;

unsigned thread_id;
size_t thread_arr_begin, thread_arr_end;
int *thread_arr;
#pragma omp threadprivate(thread_id, thread_arr_begin, thread_arr_end, thread_arr)


//xor-swap
#define swap(a,b) a^=b;b^=a;a^=b


int * rand_int_arr(size_t len)
{
    int *arr = malloc(len * sizeof(int));
    for (size_t i = 0; i < len; ++i) {
        arr[i] = rand() % 100;
    }

    return arr;
}

void print_array(int *arr, size_t len)
{
    int *end = arr + len;
    while (arr != end)
    {
        printf("%d ", *arr);
        ++arr;
    }
    printf("\n");
}


void odd_even_sort(int *arr, size_t len)
{
    size_t iters = len;
    --len;

    char has_changed = 2;
    char is_odd = 1;

    while (has_changed && iters)
    {
        --has_changed;
        for (size_t i = is_odd; i < len; i += 2)
        {
            if (arr[i+1] < arr[i]) {
                swap(arr[i], arr[i+1]);
                has_changed = 1;
            }
        }

        is_odd = !is_odd;
        --iters;
    }
}

void odd_even_sort_p(int *arr, size_t len)
{
    size_t iters = len;
    --len;

    char has_changed = 2;
    char is_odd = 1;

    size_t delta = (len + thread_num - 1) / thread_num;
    if (delta % 2) {
        ++delta;
    }
    
    #pragma omp parallel
    {
        unsigned thread_id = omp_get_thread_num();
        thread_arr_begin = thread_id * delta;
        thread_arr_end = thread_arr_begin + delta;

        if (len < thread_arr_end) {
            thread_arr_end = len;
        }
        if (thread_arr_end < thread_arr_begin) {
            thread_arr_begin = thread_arr_end;
        }
    }

    while (has_changed && iters)
    {
        --has_changed;

        #pragma omp parallel
        {
            char thread_has_changed = 0;

            for (size_t i = thread_arr_begin + is_odd; i < thread_arr_end; i += 2)
            {
                if (arr[i+1] < arr[i]) {
                    swap(arr[i], arr[i+1]);
                    thread_has_changed = 1;
                }
            }

            if (thread_has_changed) {
                has_changed = 1;
            }
        }

        is_odd = !is_odd;
        --iters;
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

int main(int argc, const char * argv[])
{
    /*
     OpenMP task5:
     odd-even sort
     */

   omp_set_num_threads(thread_num);

    size_t len;
    printf("enter size: ");
    scanf("%zu", &len);

    int *a = rand_int_arr(len);
    int *a_p = malloc(len * sizeof(int));
    memcpy(a_p, a, len * sizeof(int));

    double t1, t2;

    t1 = omp_get_wtime();
    odd_even_sort(a, len);
    t2 = omp_get_wtime();
    double seq_time = t2 - t1;

    t1 = omp_get_wtime();
    odd_even_sort_p(a_p, len);
    t2 = omp_get_wtime();
    double par_time = t2 - t1;

    printf("sequential: %s, %f sec\n", (is_sorted(a, len) ? "OK" : "BAD!"), seq_time);
    printf("parallel: %s, %f sec\n", (is_sorted(a_p, len) ? "OK" : "BAD!"), par_time);
    printf("SPEED-UP: %f\n", seq_time/par_time);

    // print_array(a, len);
    // print_array(a_pc, len);

    free(a);
    free(a_p);

    return 0;
}

