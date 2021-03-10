#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//xor-swap(a,b): a^b^b = a
#define swap(a,b) a^=b;b^=a;a^=b

#define numtype unsigned

unsigned thread_num = 4;

numtype *thread_vector;
#pragma omp threadprivate(thread_vector)

struct vector
{
    numtype *el;
    unsigned l;
};

struct matrix
{
    numtype **el;
    unsigned n;
    unsigned m;
};

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


void vector_rand_init(struct vector *v, unsigned len)
{
    v->l = len;
    v->el = malloc(v->l * sizeof(numtype));
    for (unsigned i = 0; i < v->l; ++i)
    {
        v->el[i] = rand() % 100;
    }
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

void print_vector(struct vector v)
{
    printf("(\t");
    for (unsigned i = 0; i < v.l; ++i) {
        printf("%d\t", v.el[i]);
    }
    printf(")\n");
}


struct vector matrix_vector_mult_prep(struct matrix m, struct vector v)
{
    if (m.m != v.l)
    {
        fprintf(stderr, "Matrix and vector sizes are incompatible for multiplication.\n");
        exit(-1);
    }

    struct vector res = {malloc(m.n * sizeof(numtype)), m.n};
    return res;
}

void matrix_vector_mult(struct matrix m, struct vector v, struct vector res)
{
    for (unsigned i = 0; i < m.n; ++i)
    {
        res.el[i] = 0;
        for (unsigned j = 0; j < m.m; ++j) {
            res.el[i] += m.el[i][j] * v.el[j];
        }
    }
}

void matrix_vector_mult_rp(struct matrix m, struct vector v, struct vector res)
{
    #pragma omp parallel for
    for (unsigned i = 0; i < m.n; ++i)
    {
        numtype *m_row = m.el[i];

        numtype cumsum = 0;
        for (unsigned j = 0; j < m.m; ++j) {
            cumsum += m_row[j] * v.el[j];
        }
        res.el[i] = cumsum;
    }
}

void matrix_vector_mult_rpc(struct matrix m, struct vector v, struct vector res)
{
    //thread_vector used as copy of vector v
    #pragma omp parallel
    {
        thread_vector = malloc(v.l * sizeof(numtype));
        for (unsigned i = 0; i < v.l; ++i) {
            thread_vector[i] = v.el[i];
        }
    }

    #pragma omp parallel for
    for (unsigned i = 0; i < m.n; ++i)
    {
        numtype *m_row = m.el[i];

        numtype cumsum = 0;
        for (unsigned j = 0; j < m.m; ++j) {
            cumsum += m_row[j] * thread_vector[j];
        }
        res.el[i] = cumsum;
    }

    #pragma omp parallel
    {
        free(thread_vector);
    }
}

void matrix_vector_mult_cp(struct matrix m, struct vector v, struct vector res)
{
    //thread_vector used as cumsum-vector for each thread
    #pragma omp parallel for
    for (unsigned i = 0; i < res.l; ++i) {
        res.el[i] = 0;
    }
    #pragma omp parallel
    {
        thread_vector = malloc(res.l * sizeof(numtype));
        for (unsigned i = 0; i < res.l; ++i) {
            thread_vector[i] = 0;
        }
    }

    #pragma omp parallel for
    for (unsigned j = 0; j < m.m; ++j)
    {
        numtype v_el = v.el[j];
        // TODO: change matrix representation to array of cols (not array of rows)

        for (unsigned i = 0; i < m.n; ++i) {
            thread_vector[i] += m.el[i][j] * v_el;
        }
    }

    #pragma omp parallel
    {
        for (unsigned i = 0; i < res.l; ++i) {
            #pragma omp atomic
            res.el[i] += thread_vector[i];
        }
        free(thread_vector);
    }
}

void matrix_vector_mult_cpt(struct matrix mt, struct vector v, struct vector res)
{
    //thread_vector used as cumsum-vector for each thread
    #pragma omp parallel for
    for (unsigned i = 0; i < res.l; ++i) {
        res.el[i] = 0;
    }
    #pragma omp parallel
    {
        thread_vector = malloc(res.l * sizeof(numtype));
        for (unsigned i = 0; i < res.l; ++i) {
            thread_vector[i] = 0;
        }
    }

    #pragma omp parallel for
    for (unsigned j = 0; j < mt.n; ++j)
    {
        numtype v_el = v.el[j];
        numtype *m_col = mt.el[j];

        for (unsigned i = 0; i < mt.m; ++i) {
            thread_vector[i] += m_col[i] * v_el;
        }
    }

    #pragma omp parallel
    {
        for (unsigned i = 0; i < res.l; ++i) {
            #pragma omp atomic
            res.el[i] += thread_vector[i];
        }
        free(thread_vector);
    }
}


void matrix_vector_mult_bp(struct matrix m, struct vector v, struct vector res)
{
    unsigned rowlen = m.n / thread_num;
    if (!rowlen) { rowlen = 1; }
    unsigned collen = m.m / thread_num;
    if (!collen) { collen = 1; }
    unsigned rows = (m.n + rowlen - 1) / rowlen;
    unsigned cols = (m.m + collen - 1) / collen;
    unsigned blocks = rows * cols;

    #pragma omp parallel for
    for (unsigned i = 0; i < res.l; ++i) {
        res.el[i] = 0;
    }

    #pragma omp parallel for
    for (unsigned block = 0; block < blocks; ++block) {
        unsigned row = rowlen * (block % rows);
        unsigned row_end = row + rowlen;
        if (m.n < row_end) {
            row_end = m.n;
        }
        unsigned col = collen * (block / rows);
        unsigned col_end = col + collen;
        if (m.m < col_end) {
            col_end = m.m;
        }

        for (; row < row_end; ++row) {
            numtype *m_row = m.el[row];

            numtype cumsum = 0;
            for (unsigned j = col; j < col_end; ++j) {
                cumsum += m_row[j] * v.el[j];
            }
            #pragma omp atomic
            res.el[row] += cumsum;
        }
    }
}


char are_vectors_equal(struct vector v1, struct vector v2)
{
    if (v1.l != v2.l) {
        return 0;
    }

    for (unsigned i = 0; i < v1.l; ++i) 
    {
        if (v1.el[i] != v2.el[i]) {
            return 0;
        }
    }
    return 1;
}


int main(int argc, const char * argv[])
{
    /*
     OpenMP task3.1:
     compute matrix-vector multiplication:
        rp - row parallel
        rpc - row parallel (with vector copy)
        cp - column parallel
        cpt - column parallel (with transposed matrix - col matrix)
        bp - block parallel
     */

    omp_set_num_threads(thread_num);

    struct matrix m;
    printf("enter sizes {n m}: ");
    scanf("%u%u", &m.n, &m.m);
    matrix_rand_init(&m, m.n, m.m);

    struct vector v;
    vector_rand_init(&v, m.m);

    double t1, t2;

    struct vector true_mult = matrix_vector_mult_prep(m, v);
    struct vector mult = matrix_vector_mult_prep(m, v);

    t1 = omp_get_wtime();
    matrix_vector_mult(m, v, true_mult);
    t2 = omp_get_wtime();
    double mult_time = t2 - t1;

    t1 = omp_get_wtime();
    matrix_vector_mult_rp(m, v, mult);
    t2 = omp_get_wtime();
    double mult_time_rp = t2 - t1;
    if (!are_vectors_equal(true_mult, mult)) {
        mult_time_rp = -1.;
    }
    
    t1 = omp_get_wtime();
    matrix_vector_mult_rpc(m, v, mult);
    t2 = omp_get_wtime();
    double mult_time_rpc = t2 - t1;
    if (!are_vectors_equal(true_mult, mult)) {
        mult_time_rpc = -1.;
    }

    t1 = omp_get_wtime();
    matrix_vector_mult_cp(m, v, mult);
    t2 = omp_get_wtime();
    double mult_time_cp = t2 - t1;
    if (!are_vectors_equal(true_mult, mult)) {
        mult_time_cp = -1.;
    }

    t1 = omp_get_wtime();
    matrix_vector_mult_bp(m, v, mult);
    t2 = omp_get_wtime();
    double mult_time_bp = t2 - t1;
    if (!are_vectors_equal(true_mult, mult)) {
        mult_time_bp = -1.;
    }

    matrix_transpose(&m);
    t1 = omp_get_wtime();
    matrix_vector_mult_cpt(m, v, mult);
    t2 = omp_get_wtime();
    double mult_time_cpt = t2 - t1;
    if (!are_vectors_equal(true_mult, mult)) {
        mult_time_cpt = -1.;
    }

    printf("TIME - seq: %f, rp: %f, rpc: %f, cp: %f, cpt: %f, bp: %f\n",
           mult_time, mult_time_rp, mult_time_rpc, mult_time_cp, mult_time_cpt, mult_time_bp);
    printf("SPEED-UP (opt: %u) - rp: %f, rpc: %f, cp: %f, cpt: %f, bp: %f\n",
           thread_num, mult_time/mult_time_rp, mult_time/mult_time_rpc, mult_time/mult_time_cp, mult_time/mult_time_cpt, mult_time/mult_time_bp);

    free_matrix(m);
    free(v.el);
    free(true_mult.el);
    free(mult.el);

    return 0;
}
