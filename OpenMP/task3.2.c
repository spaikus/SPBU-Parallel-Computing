#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//xor-swap(a,b): a^b^b = a
#define swap(a,b) a^=b;b^=a;a^=b
#define numtype unsigned

struct matrix
{
    numtype **el;
    unsigned n;
    unsigned m;
};

void free_matrix(struct matrix m);
void matrix_transpose(struct matrix *m);
void matrix_rand_init(struct matrix *m, unsigned rows, unsigned cols);
struct matrix matrix_matrix_mult_prep(struct matrix m1, struct matrix m2t);
void matrix_matrix_mult(struct matrix m1, struct matrix m2t, struct matrix res);
void matrix_matrix_mult_sp(struct matrix m1, struct matrix m2t, struct matrix res);
void matrix_matrix_mult_bp(struct matrix m1, struct matrix m2t, struct matrix res);
char are_matrices_equal(struct matrix m1, struct matrix m2);

unsigned num_threads = 8;

int main(int argc, const char * argv[])
{
    /*
     OpenMP task3.2:
     compute matrix-matrix multiplication:
        sp - striped parallel
        bp - block parallel
     */

    int num_tests = 10;
    int test_from = 50, test_to = 250;
    int test_step = 50;
    const char *out_file_name = "task3.2_res.txt";
    
    omp_set_num_threads(num_threads);

    FILE *output = fopen(out_file_name, "w");
    fprintf(output, 
            "Время в мкс, количество потоков: %d\n"
            "| Размер | Послед. | Пар. (ленты) | Пар. (блоки) | Ускор. (ленты) | Ускор. (блоки) |\n"
            "| :---: | :---: | :---: | :---: | :---: | :---: |\n", num_threads);

    for (int test = test_from; test <= test_to; test += test_step)
    {
        double ts = .0, tsp = .0, tbp = .0;
        double time_point;

        for (int i = 0; i < num_tests; ++i)
        {
            struct matrix m1, m2;
            m1.n = m1.m = m2.m = test;
            matrix_rand_init(&m1, m1.n, m1.m);
            matrix_rand_init(&m2, m1.m, m2.m);

            matrix_transpose(&m2);
            struct matrix mult = matrix_matrix_mult_prep(m1, m2);
            double t1, t2;


            time_point = omp_get_wtime();
            matrix_matrix_mult(m1, m2, mult);
            ts += omp_get_wtime() - time_point;

            time_point = omp_get_wtime();
            matrix_matrix_mult_sp(m1, m2, mult);
            tsp += omp_get_wtime() - time_point;

            time_point = omp_get_wtime();
            matrix_matrix_mult_bp(m1, m2, mult);
            tbp += omp_get_wtime() - time_point;


            free_matrix(m1);
            free_matrix(m2);
            free_matrix(mult);
        }
        
        ts /= num_tests / 1e6;
        tsp /= num_tests / 1e6;
        tbp /= num_tests / 1e6;

        fprintf(output, "| %d | %.0f | %.0f | %.0f | %.1f | %.1f |\n",
                test, ts, tsp, tbp, ts/tsp, ts/tbp);
    }
    
    fclose(output);

    return 0;
}


void free_matrix(struct matrix m)
{
    for (unsigned i = 0; i < m.n; ++i) {
        free(m.el[i]);
    }
    free(m.el);
}

void matrix_transpose(struct matrix *m)
{
    numtype **tmp;
    tmp = malloc(m->m * sizeof(void *));
    for (unsigned i = 0; i < m->m; ++i)
    {
        tmp[i] = malloc(m->n * sizeof(numtype));
        for (unsigned j = 0; j < m->n; ++j) {
            tmp[i][j] = m->el[j][i];
        }
    }

    free_matrix(*m);
    m->el = tmp;
    swap(m->n, m->m);
}

void matrix_rand_init(struct matrix *m, unsigned rows, unsigned cols)
{
    m->n = rows;
    m->m = cols;
    m->el = malloc(rows * sizeof(void *));
    for (unsigned i = 0; i < rows; ++i)
    {
        m->el[i] = malloc(cols * sizeof(numtype));
        for (unsigned j = 0; j < cols; ++j) {
            m->el[i][j] = rand() % 100;
        }
    }
}


void print_matrix(struct matrix m)
{
    for (unsigned i = 0; i < m.n; ++i)
    {
        printf("|\t");
        for (unsigned j = 0; j < m.m; ++j) {
            printf("%d\t", m.el[i][j]);
        }
        printf("|\n");
    }
    printf("\n");
}

struct matrix matrix_matrix_mult_prep(struct matrix m1, struct matrix m2t)
{
    if (m1.m != m2t.m)
    {
        fprintf(stderr, "Matrix sizes are incompatible for multiplication.\n");
        exit(-1);
    }

    struct matrix res = {malloc(m1.n * sizeof(void *)), m1.n, m2t.n};
    for (unsigned i = 0; i < m1.n; ++i) {
        res.el[i] = malloc(m2t.n * sizeof(numtype));
    }

    return res;
}

void matrix_matrix_mult(struct matrix m1, struct matrix m2t, struct matrix res)
{
    for (unsigned i = 0; i < m1.n; ++i)
    {
        for (unsigned j = 0; j < m2t.n; ++j)
        {
            numtype cumsum = 0;
            numtype *m1_row = m1.el[i];
            numtype *m2_col = m2t.el[j];

            for (unsigned k = 0; k < m1.m; ++k) {
                cumsum += m1_row[k] * m2_col[k];
            }
            res.el[i][j] = cumsum;
        }
    }
}

void matrix_matrix_mult_sp(struct matrix m1, struct matrix m2t, struct matrix res)
{
    unsigned cells = m1.n * m2t.n;
    #pragma omp parallel for
    for (unsigned cell = 0; cell < cells; ++cell)
    {
        unsigned i = cell % m1.n;
        unsigned j = cell / m1.n;

        numtype *m1_row = m1.el[i];
        numtype *m2_col = m2t.el[j];

        numtype res_el = 0;
        for (unsigned k = 0; k < m1.m; ++k) {
            res_el += m1_row[k] * m2_col[k];
        }

        res.el[i][j] = res_el;
    }
}

void matrix_matrix_mult_bp(struct matrix m1, struct matrix m2t, struct matrix res)
{
    unsigned rowlen = m1.n / num_threads;
    if (!rowlen) { rowlen = 1; }
    unsigned midlen = m1.m / num_threads;
    if (!midlen) { midlen = 1; }
    unsigned collen = m2t.n / num_threads;
    if (!collen) { collen = 1; }
    unsigned rows = (m1.n + rowlen - 1) / rowlen;
    unsigned mids = (m1.m + midlen - 1) / midlen;
    unsigned cols = (m2t.n + collen - 1) / collen;
    unsigned res_blocks = rows * cols;
    unsigned blocks = rows * mids * cols;

    #pragma omp parallel for
    for (unsigned i = 0; i < m1.n; ++i)
    {
        numtype *res_row = res.el[i];
        for (unsigned j = 0; j < m2t.n; ++j) {
            res_row[j] = 0;
        }
    }

    #pragma omp parallel for
    for (unsigned block = 0; block < blocks; ++block)
    {
        unsigned mid = midlen * (block / res_blocks);
        unsigned row = block % res_blocks;
        unsigned col = collen * (row / rows);
        row = rowlen * (row % rows);

        unsigned rowend = row + rowlen;
        if (m1.n < rowend) {
            rowend = m1.n;
        }
        unsigned midend = mid + midlen;
        if (m1.m < midend) {
            midend = m1.m;
        }
        unsigned colend = col + collen;
        if (m2t.n < colend) {
            colend = m2t.n;
        }

        for (; row < rowend; ++row)
        {
            for (unsigned j = col; j < colend; ++j)
            {
                numtype cumsum = 0;
                numtype *m1_row = m1.el[row];
                numtype *m2_col = m2t.el[j];

                for (unsigned k = mid; k < midend; ++k) {
                    cumsum += m1_row[k] * m2_col[k];
                }

                #pragma omp atomic
                res.el[row][j] += cumsum;
            }
        }
    }
}

char are_matrices_equal(struct matrix m1, struct matrix m2)
{
    if (m1.n != m2.n || m1.m != m2.m) {
        return 0;
    }

    for (unsigned i = 0; i < m1.n; ++i)
    {
        for (unsigned j = 0; j < m1.m; ++j)
        {
            if (m1.el[i][j] != m2.el[i][j]) {
                return 0;
            }
        }
    }

    return 1;
}
