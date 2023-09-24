#include <stdio.h>
#include <stdlib.h>

int main(int argc, char* argv[]) {
    // Kernel
    double A[256];
    double B[256];
    for (int i = 0; i < 256; i++) {
        A[i] = (double)rand()/(double)(RAND_MAX);
        B[i] = (double)rand()/(double)(RAND_MAX);
    }

    double C[256];
    for (int i = 0; i < 256; i++) {
        if (A[i] > 0) {
            B[i] = A[i] + 2;
        }
        C[i] = B[i] + 1;
    }

    for (int i = 0; i < 256; i++) {
        printf("%f", C[i]);
    }

    return 0;
}