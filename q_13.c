#include <stdio.h>
#include <gsl/gsl_linalg.h>

void print_matrix(const char *name, gsl_matrix *m) {
    printf("%s:\n", name);
    for (size_t i = 0; i < m->size1; ++i) {
        for (size_t j = 0; j < m->size2; ++j) {
            printf("%8.3f ", gsl_matrix_get(m, i, j));
        }
        printf("\n");
    }
    printf("\n");
}

int main() {
    // Define matrices for all 4 systems
    double data1[] = {3, -1, 1, 3, 6, 2, 3, 3, 7};
    gsl_matrix_view A1 = gsl_matrix_view_array(data1, 3, 3);

    double data2[] = {10, -1, 0, -1, 10, -2, 0, -2, 10};
    gsl_matrix_view A2 = gsl_matrix_view_array(data2, 3, 3);

    double data3[] = {10, 5, 0, 0, 5, 10, -4, 0, 0, -4, 8, -1, 0, 0, -1, 5};
    gsl_matrix_view A3 = gsl_matrix_view_array(data3, 4, 4);

    double data4[] = {4, 1, 1, 0, 1, -1, -3, 1, 1, 0, 2, 1, 5, -1, -1, -1, -1, -1, 4, 0, 0, 2, -1, 1, 4};
    gsl_matrix_view A4 = gsl_matrix_view_array(data4, 5, 5);

    // Perform LU decomposition for all 4 systems
    gsl_permutation *p1 = gsl_permutation_alloc(3);
    gsl_permutation *p2 = gsl_permutation_alloc(3);
    gsl_permutation *p3 = gsl_permutation_alloc(4);
    gsl_permutation *p4 = gsl_permutation_alloc(5);

    int signum1, signum2, signum3, signum4;
    gsl_linalg_LU_decomp(&A1.matrix, p1, &signum1);
    gsl_linalg_LU_decomp(&A2.matrix, p2, &signum2);
    gsl_linalg_LU_decomp(&A3.matrix, p3, &signum3);
    gsl_linalg_LU_decomp(&A4.matrix, p4, &signum4);

    // Print LU decomposition results for all 4 systems
    print_matrix("Matrix A1", &A1.matrix);
    printf("Permutation matrix P1:\n");
    gsl_permutation_fprintf(stdout, p1, " %d");
    printf("\n");

    print_matrix("Matrix A2", &A2.matrix);
    printf("Permutation matrix P2:\n");
    gsl_permutation_fprintf(stdout, p2, " %d");
    printf("\n");

    print_matrix("Matrix A3", &A3.matrix);
    printf("Permutation matrix P3:\n");
    gsl_permutation_fprintf(stdout, p3, " %d");
    printf("\n");

    print_matrix("Matrix A4", &A4.matrix);
    printf("Permutation matrix P4:\n");
    gsl_permutation_fprintf(stdout, p4, " %d");
    printf("\n");

    // Free allocated memory
    gsl_permutation_free(p1);
    gsl_permutation_free(p2);
    gsl_permutation_free(p3);
    gsl_permutation_free(p4);

    return 0;
}
