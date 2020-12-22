#include <stdio.h>
#include <stdlib.h>

int main (int argc, char** argv) {
    int i;
#pragma omp target
#pragma omp parallel for
    for (i = 0; i < 10; i++) {
        printf("Test.\n");
    }

    return 0;
}
