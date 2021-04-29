#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

static unsigned num_threads = 8;

void odd_even_sort(int *arr, size_t len);
void odd_even_sort_p(int *arr, size_t len);
int * rand_int_arr(size_t len);

int main(int argc, const char * argv[])
{
    /*
     OpenMP task5:
     odd-even sort
     */

    int num_tests = 10;
    int test_from = 4000, test_to = 20000;
    int test_step = 4000;
    const char *out_file_name = "task5_res.txt";
    
    omp_set_num_threads(num_threads);


    FILE *output = fopen(out_file_name, "w");
    fprintf(output, 
            "| Размер | Последовательный | Параллельный | Ускорение (8 потоков) |\n"
            "| --- | :---: | :---: | :---: |\n");

    for (int test = test_from; test <= test_to; test += test_step)
    {
        double ts = .0, tp = .0;
        double time_point;

        for (int i = 0; i < num_tests; ++i)
        {
            int *a = rand_int_arr(test);
            int *a_p = malloc(test * sizeof(int));
            memcpy(a_p, a, test * sizeof(int));

            time_point = omp_get_wtime();
            odd_even_sort(a, test);
            ts += omp_get_wtime() - time_point;

            time_point = omp_get_wtime();
            odd_even_sort_p(a_p, test);
            tp += omp_get_wtime() - time_point;

            free(a);
            free(a_p);
        }
        
        ts /= num_tests;
        tp /= num_tests;
        double speedup = ts / tp;

        fprintf(output, "| %d | %f | %f | %f |\n", test, ts, tp, speedup);
    }
    
    fclose(output);

    return 0;
}

unsigned thread_id;
size_t thread_arr_begin, thread_arr_end;
#pragma omp threadprivate(thread_id, thread_arr_begin, thread_arr_end)

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

    size_t delta = (len + num_threads - 1) / num_threads;
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

