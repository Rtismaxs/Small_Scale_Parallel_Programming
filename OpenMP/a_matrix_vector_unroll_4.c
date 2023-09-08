// 
// Author: Mathis Cadier mathis.cadier.430@cranfield.ac.uk
//

// Computes matrix-vector product between a sparse matrix and a vector
// Sparse Matrix A is stored in 2 formats : CSR and ELLPACK
// We generate randomly a vector x 

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <omp.h> // For using OpenMP API
#include "mmio.h" // For reading matrix.mtx
#include "wtime.h" // For calculating time

// Define the number of time, we will execute our OpenMP code
const int ntimes=1000;

// Return the max between two integers
inline int max ( int a, int b ) { return a > b ? a : b; }



//////////////////////////////////////////////////////
//                                                  //
// --------------- CSR FUNCTIONS ------------------ //
//                                                  //
//////////////////////////////////////////////////////

// Simple CPU implementation of matrix-vector product for CSR
void CSR_MatrixVector(int rows, int cols, const int* IRP, const int* J, double* val, const double* x, double* restrict y) 
{  
  for (int i = 0; i < rows; ++i) {
    double t=0.0;
    for (int j = IRP[i]-1; j < IRP[i+1]-1; ++j) {
      t += val[j] * x[J[j]-1];
    }
    y[i] = t;
  }
}

// OpenMP implementation of matrix-vector product for CSR
// where we are doing a loop unrolling on the variable i (rows) by a factor of 4
void CSR_UnrollMatrixVector(int rows, int cols, const int* IRP, const int* J, const double* val, const double* x, double* restrict y) 
{
  int i,j;
#pragma omp parallel for shared(x,y,val,IRP,J) private(i,j)
  // Unrolling on i by a factor of 4
  for (i = 0; i < rows - rows%4; i += 4) {
    
    // Initializing our indexes
    int j0 = IRP[i] - 1, end0 = IRP[i + 1] - 1;
    int j1 = IRP[i + 1] - 1, end1 = IRP[i + 2] - 1;
    int j2 = IRP[i + 2] - 1, end2 = IRP[i + 3] - 1;
    int j3 = IRP[i + 3] - 1, end3 = IRP[i + 4] - 1;
    
    double t0 = val[j0] * x[J[j0]-1];
    double t1 = val[j1] * x[J[j1]-1];
    double t2 = val[j2] * x[J[j2]-1];
    double t3 = val[j3] * x[J[j3]-1];
    
    for (int j = j0 + 1; j < end0; ++j) {
      t0 += val[j] * x[J[j]-1];
    }
    
    for (int j = j1 + 1; j < end1; ++j) {
      t1 += val[j] * x[J[j]-1];
    }
    
    for (int j = j2 + 1; j < end2; ++j) {
      t2 += val[j] * x[J[j]-1];
    }
    
    for (int j = j3 + 1; j < end3; ++j) {
      t3 += val[j] * x[J[j]-1];
    }
    
    y[i+0] = t0;
    y[i+1] = t1;
    y[i+2] = t2;
    y[i+3] = t3;

  }
  
  // Processing the remaining rows
  for (i = rows - rows%4; i < rows; i++) {
    double t=0.0;
    for (j = IRP[i]-1; j < IRP[i+1]-1; ++j) {
      t += val[j] * x[J[j]-1];
    }
    y[i] = t;
  }
}



///////////////////////////////////////////////////////
//                                                   //
// ------------- ELLPACK FUNCTIONS ----------------- //
//                                                   //
///////////////////////////////////////////////////////

// Simple CPU implementation of matrix-vector product for ELLPACK
void ELL_MatrixVector(int rows, int cols, int MAXNZ, const int* JA, const double* AS, const double* x, double* restrict y) 
{  
  for (int i = 0; i < rows; ++i) {
    double t=0.0;
    for (int j = 0; j < MAXNZ; ++j) {
      int idx = i * MAXNZ + j;
      t += AS[idx] * x[JA[idx]-1];
    }
    y[i] = t;
  }
}


// OpenMP implementation of matrix-vector product for ELLPACK
// where we are doing a loop unrolling on the variable i (rows) by a factor of 4
void ELL_UnrollMatrixVector(int rows, int cols, int MAXNZ, const int* JA, const double* AS, const double* x, double* restrict y) 
{
  int i,j,idx;
#pragma omp parallel for shared(x,y,AS,JA) private(i,j,idx)
  // Unrolling on i by a factor of 4
  for (i = 0; i < rows - rows%4; i += 4) {
    
    // Initializing our indexes
    double t0 = AS[(i+0) * MAXNZ] * x[JA[(i+0) * MAXNZ]-1];
    double t1 = AS[(i+1) * MAXNZ] * x[JA[(i+1) * MAXNZ]-1];
    double t2 = AS[(i+2) * MAXNZ] * x[JA[(i+2) * MAXNZ]-1];
    double t3 = AS[(i+3) * MAXNZ] * x[JA[(i+3) * MAXNZ]-1];    

    for (j = 1; j < MAXNZ; j++) {
      t0 += AS[(i+0) * MAXNZ + j] * x[JA[(i+0) * MAXNZ + j]-1];
      t1 += AS[(i+1) * MAXNZ + j] * x[JA[(i+1) * MAXNZ + j]-1];
      t2 += AS[(i+2) * MAXNZ + j] * x[JA[(i+2) * MAXNZ + j]-1];
      t3 += AS[(i+3) * MAXNZ + j] * x[JA[(i+3) * MAXNZ + j]-1];

    }
    y[i+0] = t0;
    y[i+1] = t1;
    y[i+2] = t2;
    y[i+3] = t3;

  }
  
  // Processing the remaining rows
  for (i = rows - rows%4; i < rows; i++) {
    double t=0.0;
    for (j = 0; j < MAXNZ; ++j) {
      int idx = i * MAXNZ + j;
      t += AS[idx] * x[JA[idx]-1];
    }
    y[i] = t;
  }
}



////////////////////////////////////////////////////
//                                                //
// -------------- MAIN FUNCTION ----------------- //
//                                                //
////////////////////////////////////////////////////

int main(int argc, char *argv[])
{
       
    // Declaring variables
    MM_typecode matcode;
    int M, N, nz, MAXNZ;   
    int i, *I, *J, *IRP, *IRP_v, *JA_csr, *JA_ell;
    double *val, *AS_csr, *AS_ell;
    
    // Reading the MatrixMarket file
    if (argc < 2)
	  {
    		fprintf(stderr, "Usage: %s [martix-market-filename]\n", argv[0]);
    		exit(1);
	  }
    
    if (mm_read_mtx_crd(argv[1], &(M), &(N), &(nz), &(I), &(J), &(val),  &(matcode)) !=0)
    {
        fprintf(stderr, "Error in reading the matrix !\n");
        exit(1);
    }
          
    fprintf(stdout, "Test case: %d x %d with %d NZ \n", M, N, nz);
    
    
    /**************************************/
    /* reseve memory for x and y matrices */
    /**************************************/
    
    double* x = (double*) malloc(sizeof(double)*M );
    
    double* serial_y_ell = (double*) malloc(sizeof(double)*M );
    double* serial_y_csr = (double*) malloc(sizeof(double)*M );
    double* unroll_y_ell = (double*) malloc(sizeof(double)*M );
    double* unroll_y_csr = (double*) malloc(sizeof(double)*M );
    
    // Generating randomly x (but always with the same values thanks to srand(12345))
    srand(12345);
    for (int row = 0; row < M; ++row){        
        x[row] = 100.0f * ((double) rand()) / RAND_MAX;
    }

    /**********************************/
    /* reseve memory for CSR matrices */
    /**********************************/

    JA_csr = (int *) malloc(nz * sizeof(int));
    AS_csr = (double *) malloc(nz * sizeof(double));
    
    IRP = (int *) calloc((M + 1), sizeof(int));
    IRP_v = (int *) calloc((M + 1), sizeof(int));
    
    
    /***************************/
    /* reseve memory for MAXNZ */
    /***************************/
    
    int* NZ = (int *) calloc(M, sizeof(int));    
    MAXNZ = 0;
    
    // Calculating MAXNZ for the ELLPACK format and IRP for the CSR format
    for (int i = 0; i < nz; i++) {
        IRP[I[i]+1]++;
        NZ[I[i]]++;
    }
    
    for (i = 0; i < M; i++) {
        if (NZ[i] > MAXNZ){
            MAXNZ = NZ[i];
        }
    }
    
    fprintf(stdout, "\n");
    fprintf(stdout, "MAXNZ : %d \n", MAXNZ);
    fprintf(stdout, "\n");
    
    // Calculating JA and AS for CSR format
    IRP[0] = 1;
    IRP_v[0] = 1;
    
    for (int i = 1; i <= M; i++) {
        IRP[i] += IRP[i-1];
        IRP_v[i] = IRP[i];
    }
        
    for (int i = 0; i < nz; i++){
        int row = I[i];
        int src = IRP_v[row]-1;
        JA_csr[src] = J[i]+1;  
        AS_csr[src] = val[i];
        IRP_v[row]++;        
    }
    
    
    /**********************************/
    /* reseve memory for ELL matrices */
    /**********************************/
    
    JA_ell = (int *) calloc(M * MAXNZ, sizeof(int));
    AS_ell = (double *) calloc(M * MAXNZ, sizeof(double));
    
    // Calculating JA and AS for ELLPACK format
    for (int i = 0; i <= M; i++) {
        IRP_v[i] = 0;
    }
    
    for (int i = 0; i < nz; i++){
        int row = I[i];
        int src = IRP_v[row];
        JA_ell[row * MAXNZ + src] = J[i]+1;  
        AS_ell[row * MAXNZ + src] = val[i];
        IRP_v[row]++;        
    }
    
    /***********************************/
    /* Matrix-vector product on Serial */
    /***********************************/

    CSR_MatrixVector(M, N, IRP, JA_csr, AS_csr, x, serial_y_csr);
    ELL_MatrixVector(M, N, MAXNZ, JA_ell, AS_ell, x, serial_y_ell);
    
    double t1;
    double t2;
    double maxabs;
    
    /*********************************/
    /* Matrix-vector product for CSR */
    /*********************************/
        
    double tmlt = 0.0;
    for (int try=0; try < ntimes; try ++ ) {
        t1 = wtime();
        CSR_UnrollMatrixVector(M, N, IRP, JA_csr, AS_csr, x, unroll_y_csr);
        t2 = wtime();
        tmlt += (t2-t1);
    }
    
    // Average runtime
    tmlt /= ntimes;
    double mflops = (2.0e-6)*nz/tmlt;
    
    double reldiff_csr = 0.0f;
    double diff_csr = 0.0f;
    
    // Now let's check if the results are the same for serial CSR and Unrolling 4 CSR
    for (int row = 0; row < M; ++row) {
      maxabs = max(fabs(serial_y_csr[row]),fabs(unroll_y_csr[row]));
      if (maxabs == 0.0) maxabs=1.0;
      reldiff_csr = max(reldiff_csr, fabs(serial_y_csr[row] - unroll_y_csr[row])/maxabs);
      diff_csr = max(diff_csr, fabs(serial_y_csr[row] - unroll_y_csr[row]));
    }
    
#pragma omp parallel 
    {
#pragma omp master
      {
        fprintf(stdout,"CSR Matrix-Vector product (unroll_4) of size %d x %d with %d threads: time %lf  MFLOPS %lf  Max diff CSR = %lf  Max rel diff CSR = %lf \n",M,N,omp_get_num_threads(),tmlt,mflops, diff_csr, reldiff_csr);
      }
    }
    fprintf(stdout, "\n"); 
       
    /*********************************/
    /* Matrix-vector product for ELL */
    /*********************************/
    
    tmlt = 0.0;
    for (int try=0; try < ntimes; try ++ ) {
        t1 = wtime();
        ELL_UnrollMatrixVector(M, N, MAXNZ, JA_ell, AS_ell, x, unroll_y_ell);
        t2 = wtime();
        tmlt += (t2-t1);
    }
    
    // Average runtime
    tmlt /= ntimes;
    mflops = (2.0e-6)*nz/tmlt;
    
    double reldiff_ell = 0.0f;
    double diff_ell = 0.0f;
    
    // Now let's check if the results are the same for serial ELLPACK and Unrolling 4 ELLPACK
    for (int row = 0; row < M; ++row) {
      maxabs = max(fabs(serial_y_ell[row]),fabs(unroll_y_ell[row]));
      if (maxabs == 0.0) maxabs=1.0;
      reldiff_ell = max(reldiff_ell, fabs(serial_y_ell[row] - unroll_y_ell[row])/maxabs);
      diff_ell = max(diff_ell, fabs(serial_y_ell[row] - unroll_y_ell[row]));
    }
    
    
#pragma omp parallel 
    {
#pragma omp master
      {
        fprintf(stdout,"ELLPACK Matrix-Vector product (unroll_4) of size %d x %d with %d threads: time %lf  MFLOPS %lf  Max diff ELLPACK = %lf  Max rel diff ELLPACK = %lf \n",M,N,omp_get_num_threads(),tmlt,mflops, diff_ell, reldiff_ell);
      }
    }
    fprintf(stdout, "\n");
    
    
    // Cleaning up
    free(IRP);
    free(IRP_v);
    free(JA_csr);
    free(AS_csr);
    free(JA_ell);
    free(AS_ell);
    free(I);
    free(J);
    free(val);
    free(x);
    free(serial_y_csr);
    free(serial_y_ell);
    free(unroll_y_csr);
    free(unroll_y_ell);

	return 0;
}
