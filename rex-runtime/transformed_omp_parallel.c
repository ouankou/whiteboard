#include <stdio.h>
#include <omp.h>
#include "rex_kmp.h"

void outlined_function() {
	int id = omp_get_thread_num();
	int num_threads = omp_get_num_threads();
   	printf("Hello World from thread %d of %d!\n", id, num_threads);
}

int main(int argc, char* argv[])
 {
   __kmpc_fork_call(NULL, 0, outlined_function);

   return 0;
 }

