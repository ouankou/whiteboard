#include<stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


void decompose(double *matrix, long size) {

    /* in situ LU decomposition */
    int k, row;

    //LU-decomposition based on Gaussian Elimination
    // - Arranged so that the multiplier doesn't have to be computed multiple times
    for (k = 0; k < size-1; k++) { //iterate over rows/columns for elimination
        // The "multiplier" is the factor by which a row is multiplied when
        //  being subtracted from another row.
#pragma omp parallel for 
        for (row = k + 1; row < size; row++) {
            matrix[row*size+k] /= matrix[k*size+k];
        };
#pragma omp parallel for private(row) shared(matrix)
        for (row = k + 1; row < size; row++) {
            int col = 0;
            double factor = matrix[row*size+k];
            //Eliminate entries in sub matrix (subtract rows)
            for (col = k + 1; col < size; col++) { //column
                matrix[row*size+col] = matrix[row*size+col] - factor*matrix[k*size+col];
            }
            matrix[row*size+k] = factor;
        }
    }
}

