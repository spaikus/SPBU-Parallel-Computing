#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <omp.h>

#define numtype int

int main()
{
    /*
    OpenMP task2:
    find max of min row elements
    */
   
    unsigned n;
    printf("enter matrix size: ");
    scanf("%u", &n);

    unsigned matrix_size = n*n;
    numtype *matrix = malloc(matrix_size * sizeof(numtype));
    #pragma omp parallel for
    for (unsigned i = 0; i < matrix_size; ++i)
    {
        matrix[i] = rand() % 100;
    }

    // print matrix
    if (n < 10)
    {
        unsigned i = 0;
        unsigned rowend;
        while (i < matrix_size)
        {
            rowend = i + n;
            while (i < rowend)
            {
                printf("%d ", matrix[i]);
                ++i;
            }   printf("\n");
        }
    }

    int max_of_mins = INT_MIN;
    #pragma omp parallel for reduction(max: max_of_mins)
    for (unsigned i = 0; i < n; ++i)
    {
        unsigned j = i * n;
        unsigned endrow = j + n;

        numtype rowmin = matrix[j];
        while (++j < endrow)
        {
            if (matrix[j] < rowmin) {
                rowmin = matrix[j];
            }
        }

        if (max_of_mins < rowmin) {
            max_of_mins = rowmin;
        }

        printf("%u row: %d\n", i, rowmin);
    }

    printf("max of row mins: %d\n", max_of_mins);
    free(matrix);

    return 0;
}
