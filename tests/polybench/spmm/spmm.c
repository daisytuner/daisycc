#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <math.h>

/* Include polybench common header. */
#include <polybench.h>

/* Include benchmark-specific header. */
#include "spmm.h"


/* Array initialization. */
static
void init_array(int ni, int nj, int nk,
		DATA_TYPE POLYBENCH_1D(A_val,NI,ni),
		int POLYBENCH_1D(A_row,NI+1,ni+1),
		int POLYBENCH_1D(A_col,NK,nk),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < NI; i++)
    A_val[i] = (DATA_TYPE) (i+1);

  for (i = 0; i < NI + 1; i++)
    A_row[i] = i;

  for (i = 0; i < NK; i++)
    A_col[i] = i;

  for (i = 0; i < nk; i++)
    for (j = 0; j < nj; j++)
      B[i][j] = (DATA_TYPE) ((i+1)*(j+1) % nj) / nj;
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++)
      C[i][j] = (DATA_TYPE) 0.0;
}


/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j;

  POLYBENCH_DUMP_START;
  POLYBENCH_DUMP_BEGIN("C");
  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	if ((i * nj + j) % 20 == 0) fprintf (POLYBENCH_DUMP_TARGET, "\n");
	fprintf (POLYBENCH_DUMP_TARGET, DATA_PRINTF_MODIFIER, C[i][j]);
    }
  POLYBENCH_DUMP_END("C");
  POLYBENCH_DUMP_FINISH;
}


/* Main computational kernel. The whole function will be timed,
   including the call and return. */
static
void kernel_spmm(int ni, int nj, int nk,
		DATA_TYPE POLYBENCH_1D(A_val,NI,ni),
		int POLYBENCH_1D(A_row,NI + 1,ni + 1),
		int POLYBENCH_1D(A_col,NJ,nj),
		DATA_TYPE POLYBENCH_2D(B,NK,NJ,nk,nj),
		DATA_TYPE POLYBENCH_2D(C,NI,NJ,ni,nj))
{
  int i, j, k;

#pragma scop
    for (i = 0; i < _PB_NI; i++)
	    for (j = 0; j < _PB_NJ; ++j)
        for (k = A_row[i]; k < A_row[i + 1]; k++)
	        C[i][j] += A_val[k] * B[A_col[k]][j];
#pragma endscop
}


int main(int argc, char** argv)
{
  /* Retrieve problem size. */
  int ni = NI;
  int nj = NJ;
  int nk = NK;

  /* Variable declaration/allocation. */
  POLYBENCH_1D_ARRAY_DECL(A_val,DATA_TYPE,NI,ni);
  POLYBENCH_1D_ARRAY_DECL(A_row,int,NI+1,ni+1);
  POLYBENCH_1D_ARRAY_DECL(A_col,int,NK,nk);
  POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NK,NJ,nk,nj);
  POLYBENCH_2D_ARRAY_DECL(C,DATA_TYPE,NI,NJ,ni,nj);

  /* Initialize array(s). */
  init_array (ni, nj, nk,
	      POLYBENCH_ARRAY(A_val),
	      POLYBENCH_ARRAY(A_row),
	      POLYBENCH_ARRAY(A_col),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C));

  /* Start timer. */
  polybench_start_instruments;

  /* Run kernel. */
  kernel_spmm (ni, nj, nk,
	      POLYBENCH_ARRAY(A_val),
	      POLYBENCH_ARRAY(A_row),
	      POLYBENCH_ARRAY(A_col),
	      POLYBENCH_ARRAY(B),
	      POLYBENCH_ARRAY(C));

  /* Stop and print timer. */
  polybench_stop_instruments;
  polybench_print_instruments;

  /* Prevent dead-code elimination. All live-out data must be printed
     by the function call in argument. */
  polybench_prevent_dce(print_array(ni, nj,  POLYBENCH_ARRAY(C)));

  /* Be clean. */
  POLYBENCH_FREE_ARRAY(A_val);
  POLYBENCH_FREE_ARRAY(A_row);
  POLYBENCH_FREE_ARRAY(A_col);
  POLYBENCH_FREE_ARRAY(B);
  POLYBENCH_FREE_ARRAY(C);

  return 0;
}
