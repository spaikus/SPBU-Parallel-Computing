#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define is_master(id) !id
#define MASTER 0

static int id;
static int num_threads;

struct vector
{
    double *el;
    unsigned n;
};

struct matrix
{
    double **el;
    unsigned n;
};

void vector_rand_init(struct vector *v, int n);
void matrix_rand_init(struct matrix *m, int n);
void free_matrix(struct matrix m);
void print_vector(struct vector v);
void print_matrix(struct matrix m);

void LU_decomposition(struct matrix m, int *permutation);
void LU_decomposition_par_m(struct matrix m, int *permutation);
void LU_decomposition_par();
void LU_solver(struct matrix m, struct vector v, int *permutation);


int main(int argc, const char * argv[])
{
    /*
     MPI task3
     LU decomposition
     */

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    MPI_Comm_size(MPI_COMM_WORLD, &num_threads);


    double time_point;
    double ts, tp;

    if (is_master(id))
    {
        int n=1000;
        printf("enter size: ");
        fflush(stdout);
        scanf("%d", &n);

        struct matrix m;
        matrix_rand_init(&m, n);

        struct matrix mp;
        mp.n = m.n;
        mp.el = malloc(n * sizeof(void *));
        for (int i = 0; i < n; ++i)
        {
            mp.el[i] = malloc(n * sizeof(double));
            memcpy(mp.el[i], m.el[i], n * sizeof(double));
        }

        struct vector v;
        vector_rand_init(&v, n);

        struct vector vp;
        vp.n = v.n;
        vp.el = malloc(n * sizeof(double));
        memcpy(vp.el, v.el, n * sizeof(double));



        int *permutation = malloc(m.n * sizeof(int));

        time_point = MPI_Wtime();
        LU_decomposition(m, permutation);
        ts = MPI_Wtime() - time_point;


        int *permutation_p = malloc(mp.n * sizeof(int));

        time_point = MPI_Wtime();
        LU_decomposition_par_m(mp, permutation_p);
        tp = MPI_Wtime() - time_point;


        printf("LU_seq: %f, LU_par: %f\nSPEED-UP: %f\n", ts, tp, ts/tp);


        LU_solver(m, v, permutation);


        free_matrix(m);
        free_matrix(mp);
        free(v.el);
        free(vp.el);
        free(permutation);
        free(permutation_p);
    }
    else
    {
        LU_decomposition_par();
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

void doubleswap(double *a, double *b)
{
    double tmp = *a;
    *a = *b;
    *b = tmp;
}


void vector_rand_init(struct vector *v, int n)
{
    v->n = n;
    v->el = malloc(v->n * sizeof(double));
    for (int i = 0; i < v->n; ++i)
    {
        v->el[i] = (double)(rand() % 10000) / 100.0;
    }
}

void matrix_rand_init(struct matrix *m, int n)
{
    m->n = n;
    m->el = malloc(n * sizeof(void *));

    for (int i = 0; i < n; ++i)
    {
        m->el[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; ++j)
        {
            m->el[i][j] = (double)(rand() % 10000) / 100.0;
        }
    }
}

void free_matrix(struct matrix m)
{
    for (unsigned i = 0; i < m.n; ++i)
    {
        free(m.el[i]);
    }
    free(m.el);
}

void print_matrix(struct matrix m)
{
    for (int i = 0; i < m.n; ++i)
    {
        printf("|%f", *m.el[i]);
        for (int j = 1; j < m.n; ++j)
        {
            printf("\t%f", m.el[i][j]);
        }
        printf("|\n");
    }
}

void print_vector(struct vector v)
{
    printf("(%f", *v.el);
    for (int i = 1; i < v.n; ++i)
    {
        printf("\t%f", v.el[i]);
    }
    printf(")\n");
}


void LU_decomposition(struct matrix m, int *permutation)
{
    for (int k = 0; k < m.n; ++k)
    {
        // find pivot - max element in column
        {
            int row = k;

            for (int i = k + 1; i < m.n; ++i)
            {
                if (fabs(m.el[row][k]) < fabs(m.el[i][k]))
                {
                    row = i;
                }
            }

            permutation[k] = row;
            pntswap(&m.el[k], &m.el[row]);
        }

        // Gaussian elimination step
        for (int i = k + 1; i < m.n; ++i)
        {
            double c = m.el[i][k] = m.el[i][k] / m.el[k][k];

            for (int j = k + 1; j < m.n; ++j)
            {
                m.el[i][j] -= c * m.el[k][j];
            }
        }
    }
}


void LU_decomposition_par_m(struct matrix m, int *permutation)
{
    MPI_Bcast(&m.n, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    for (int i = 0; i < m.n; ++i)
    {
        MPI_Bcast(m.el[i], m.n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }

    int order = 0;
    double *coefficients = malloc(m.n * sizeof(double));

    for (int k = 1; k < m.n; ++k, ++order)
    {
        int obs = k - 1;

        if (order == num_threads)
        {
            order = 0;
        }

        // find pivot - max element in column
        {
            int row = obs;

            if (!order)
            {
                for (int i = k; i < m.n; ++i)
                {
                    if (fabs(m.el[row][obs]) < fabs(m.el[i][obs]))
                    {
                        row = i;
                    }
                }
            }

            MPI_Bcast(&row, 1, MPI_INT, order, MPI_COMM_WORLD);
            permutation[obs] = row;
            pntswap(&m.el[obs], &m.el[row]);

            if (!order)
            {
                for (int i = k; i < m.n; ++i)
                {
                    coefficients[i] = m.el[i][obs] = m.el[i][obs] / m.el[obs][obs];
                }
            }

            MPI_Bcast(coefficients + k, m.n - k, MPI_DOUBLE, order, MPI_COMM_WORLD);
        }


        // Gaussian elimination step
        int col = k - id;
        if (col < 0)
        {
            col = id;
        }
        else
        {
            col %= num_threads;
            if (!col)
            {
                col = k;
            }
            else
            {
                col = k + num_threads - col;
            }
        }

        for (int i = k; i < m.n; ++i)
        {
            double c = coefficients[i];
            for (int j = col; j < m.n; j += num_threads)
            {
                m.el[i][j] -= c * m.el[obs][j];
            }
        }
    }

    order = 1;
    for (int j = 1; j < m.n; ++j, ++order)
    {
        if (order == num_threads)
        {
            order = 0;
            continue;
        }

        MPI_Recv(coefficients, m.n, MPI_DOUBLE, order, MASTER, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < m.n; ++i)
        {
            m.el[i][j] = coefficients[i];
        }
    }

    free(coefficients);
}

void LU_decomposition_par()
{
    struct matrix m;
    MPI_Bcast(&m.n, 1, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

    m.el = malloc(m.n * sizeof(void *));
    for (int i = 0; i < m.n; ++i)
    {
        m.el[i] = malloc(m.n * sizeof(double));
        MPI_Bcast(m.el[i], m.n, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);
    }


    int order = 0;
    double *coefficients = malloc(m.n * sizeof(double));

    for (int k = 1; k < m.n; ++k, ++order)
    {
        int obs = k - 1;

        if (order == num_threads)
        {
            order = 0;
        }

        // find pivot - max element in column
        {
            int row = obs;

            if (order == id)
            {
                for (int i = k; i < m.n; ++i)
                {
                    if (fabs(m.el[row][obs]) < fabs(m.el[i][obs]))
                    {
                        row = i;
                    }
                }
            }

            MPI_Bcast(&row, 1, MPI_INT, order, MPI_COMM_WORLD);
            pntswap(&m.el[obs], &m.el[row]);

            if (order == id)
            {
                for (int i = k; i < m.n; ++i)
                {
                    coefficients[i] = m.el[i][obs] = m.el[i][obs] / m.el[obs][obs];
                }
            }

            MPI_Bcast(coefficients + k, m.n - k, MPI_DOUBLE, order, MPI_COMM_WORLD);
        }


        // Gaussian elimination step
        int col = k - id;
        if (col < 0)
        {
            col = id;
        }
        else
        {
            col %= num_threads;
            if (!col)
            {
                col = k;
            }
            else
            {
                col = k + num_threads - col;
            }
        }

        for (int i = k; i < m.n; ++i)
        {
            double c = coefficients[i];
            for (int j = col; j < m.n; j += num_threads)
            {
                m.el[i][j] -= c * m.el[obs][j];
            }
        }
    }

    for (int j = id; j < m.n; j += num_threads)
    {
        for (int i = 0; i < m.n; ++i)
        {
            coefficients[i] = m.el[i][j];
        }

        MPI_Send(coefficients, m.n, MPI_DOUBLE, MASTER, MASTER, MPI_COMM_WORLD);
    }

    free(coefficients);
    free_matrix(m);
}


void LU_solver(struct matrix m, struct vector v, int *permutation)
{
    for (int i = 0; i < v.n; ++i)
    {
        doubleswap(&v.el[i], &v.el[permutation[i]]);
    }

    int k;
    for (k = 0; k < v.n; ++k)
    {
        for (int i = 0; i < k; ++i)
        {
            v.el[k] -= m.el[k][i] * v.el[i];
        }
    }

    k = v.n;
    while (k)
    {
        --k;

        for (int i = v.n - 1; k < i; --i)
        {
            v.el[k] -= m.el[k][i] * v.el[i];
        }
        v.el[k] /= m.el[k][k];
    }
}