#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "softmax.h"

#define MAX(a,b) ((a) > (b) ? (a) : (b))

/* Array initialization. */
static
void init_array(int n, int m,
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m),
    DATA_TYPE POLYBENCH_1D(norm,N,n),
    DATA_TYPE POLYBENCH_1D(maxi,N,n))
{
  int i, j;

  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      A[i][j] = (DATA_TYPE) ((i*j+1) % n) / m;
  
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++)
      B[i][j] = (DATA_TYPE) 0.0;
  
  for (i = 0; i < n; i++)
    norm[i] = (DATA_TYPE) 0.0;

  for (i = 0; i < n; i++)
    maxi[i] = (DATA_TYPE) -1.0e12;

}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int n, int m, DATA_TYPE POLYBENCH_2D(B,M,M,m,m))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("B");
  for (i = 0; i < n; i++)
    for (j = 0; j < m; j++) {
	if ((i * m + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, B[i][j]);
    }
  POLYBENCH_DUMP_END("B");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_softmax(int n, int m,
		DATA_TYPE POLYBENCH_2D(A,N,M,n,m),
		DATA_TYPE POLYBENCH_2D(B,N,M,n,m),
    DATA_TYPE POLYBENCH_1D(norm,N,n),
    DATA_TYPE POLYBENCH_1D(maxi,N,n))
{
  int i, j;

#pragma scop
    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++)
	      maxi[i] = MAX(A[i][j], maxi[i]);

    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++)
	      B[i][j] = EXP_FUN(A[i][j] - maxi[i]);
    
    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++)
        norm[i] += B[i][j];

    for (i = 0; i < _PB_N; i++)
      for (j = 0; j < _PB_M; j++)
	      B[i][j] = B[i][j] / norm[i];
    
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int n = N;
  int m = M;

  /* Variable declaration/allocation. */
  POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,N,M,n,m);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,M,M,n,m);
  POLYBENCH_1D_ARRAY_DECL(norm,DATA_TYPE,N,n);
  POLYBENCH_1D_ARRAY_DECL(maxi,DATA_TYPE,N,n);

  /* Initialize array(s). */
  init_array (n, m, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(norm), POLYBENCH_ARRAY(maxi));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_softmax (n, m,
	      POLYBENCH_ARRAY(A),
	      POLYBENCH_ARRAY(B),
        POLYBENCH_ARRAY(norm),
        POLYBENCH_ARRAY(maxi));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(n, m,  POLYBENCH_ARRAY(B)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A);
  POLYBENCH_FREE_ARRAY(B);

  return 0;
}
