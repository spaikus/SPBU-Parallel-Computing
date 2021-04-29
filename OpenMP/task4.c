#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

static unsigned num_threads = 8;

#define hashtype unsigned
static const hashtype p = UCHAR_MAX + 1;
static hashtype pn;

//unsigned char *thread_string;
//#pragma omp threadprivate(thread_string)

void pn_init(size_t len)
{
    pn = 1;
    while (--len) {
        pn *= p;
    }
}

hashtype hash_string(const unsigned char *str, size_t len)
{
    //function treats len as true length of string
    //it doesn't check for \0 symbol

    // ∑ str[len-1-i] * p^i
    hashtype strhash = 0;

    const unsigned char *end = str + len;
    while (str != end) 
    {
        strhash = strhash * p + (hashtype)*str;
        ++str;
    }

    return strhash;
}

hashtype hash_string_step(const unsigned char *str, size_t len, hashtype strhash)
{
    //function treats len as true length of current and next (shifted) string
    return (strhash - pn * (*str)) * p + str[len];
}

void find_substr(
                 const char *substr, size_t sublen,
                 const char *string,
                 char *is_substr, size_t indlen
                 )
{
    hashtype subhash = hash_string((const unsigned char *)substr, sublen);
    hashtype strhash = hash_string((const unsigned char *)string, sublen);

    for (size_t i = 0; i < indlen; ++i)
    {
        if (subhash != strhash || strncmp(substr, string, sublen)) {
            *is_substr = 'n';
        } else {
            *is_substr = 'Y';
        }
        strhash = hash_string_step((const unsigned char *)string, sublen, strhash);
        ++string;
        ++is_substr;
    }
    *is_substr = '\0';
}

void find_substr_p(
                   const char *substr, size_t sublen,
                   const char *string,
                   char *is_substr, size_t indlen
                   )
{
    hashtype subhash = hash_string((const unsigned char *)substr, sublen);
    size_t delta = (indlen + num_threads - 1) / num_threads;

    #pragma omp parallel firstprivate(string, is_substr)
    {
        unsigned thread_id = omp_get_thread_num();
        size_t lind = thread_id * delta;
        size_t rind = lind + delta;

        if (indlen < rind) {
            rind = indlen;
        }
        
        if (!(rind < lind)) 
        {
            string += lind;
            is_substr += lind;
            hashtype strhash = hash_string((const unsigned char *)string, sublen);

            while (lind < rind) 
            {
                if (subhash != strhash || strncmp(substr, string, sublen)) {
                    *is_substr = 'n';
                } else {
                    *is_substr = 'Y';
                }
                strhash = hash_string_step((const unsigned char *)string, sublen, strhash);
                ++string;
                ++is_substr;
                ++lind;
            }
        }
    }

    is_substr[indlen] = '\0';
}

char * rand_str(size_t len)
{
    char *string = malloc((len + 1) * sizeof(char));
    for (size_t i = 0; i < len; ++i) {
        string[i] = 60 + rand() % 30;
    }
    string[len] = '\0';

    return string;
}

int main(int argc, const char * argv[])
{
    /*
     OpenMP task4:
     find substring
     */

    omp_set_num_threads(num_threads);

    size_t sublen, len;
    printf("enter lengths {substring string}: ");
    scanf("%zu%zu", &sublen, &len);

    char *substr = rand_str(sublen);
    char *string = rand_str(len);

    size_t indlen = len - sublen + 1;

    char *is_substr = malloc((indlen + 1) * sizeof(char));
    char *is_substr_p = malloc((indlen + 1) * sizeof(char));

    pn_init(sublen);
    double t1, t2;


    t1 = omp_get_wtime();
    find_substr(substr, sublen, string, is_substr, indlen);
    t2 = omp_get_wtime();
    double seq_time = t2 - t1;
    printf("sequential %f sec\n", seq_time);

    t1 = omp_get_wtime();
    find_substr_p(substr, sublen, string, is_substr_p, indlen);
    t2 = omp_get_wtime();
    double par_time = t2 - t1;
    printf("parallel %f sec\n", par_time);

    if (strcmp(is_substr, is_substr_p)) {
        printf("parallel and sequential results are different\n");
    } else {
        printf("SPEED-UP: %f\n", seq_time/par_time);
    }

    free(substr);    
    free(string);
    free(is_substr);
    free(is_substr_p);

    return 0;
}

