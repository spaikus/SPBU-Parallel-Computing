#include "stdio.h"
#include "omp.h"

int main()
{
    int n;
    printf("enter number of sum iterations: ");
    scanf("%d", &n);

    // sequential
    {
        double dx = 1.0 / (double)n;
        double integral = 0.0;

        for (int i = 0; i < n; ++i)
        {
            double x = dx * (double)i;
            integral += 4.0 / (1.0 + x*x);
        }

        integral *= dx;
        printf("seq | pi is about %f\n", integral);
    }

    //parallel (non-stable)
    {
        double dx = 1.0 / (double)n;
        double integral = 0.0;
        #pragma omp parallel for
        for (int i = 0; i < n; ++i)
        {
            double x = dx * (double)i;
            integral += 4.0 / (1.0 + x*x);
            
            printf("thread %d, int_sum %f\n", omp_get_thread_num(), integral);
        }

        integral *= dx;
        printf("!!parallel!! | pi is about %f\n", integral);
    }

    //parallel
    {
        double dx = 1.0 / (double)n;
        double integral = 0.0;
        #pragma omp parallel for reduction(+: integral)
        for (int i = 0; i < n; ++i)
        {
            double x = dx * (double)i;
            integral += 4.0 / (1.0 + x*x);
        }

        integral *= dx;
        printf("parallel | pi is about %f\n", integral);
    }
	
	return 0;
}
