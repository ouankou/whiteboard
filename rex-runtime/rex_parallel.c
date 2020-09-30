#include <assert.h>
#include "rex_kmp.h"
#include <stdio.h>

void outlined_function_7(int* global_id, int* bound_id, int* i, int* j) {
    int __i = *i;
    assert(__i == 100);
    assert(*j == 77);
}

void outlined_function_12(int* global_id, int* bound_id, int* j) {
    int i;
    assert(i != 100);
}

int main(void)
{
  int i = 100;
  int j = 77;

  __kmpc_fork_call(NULL, 2, outlined_function_7, &i, &j);

  __kmpc_fork_call(NULL, 1, outlined_function_12, &j);

  return 0;
}
