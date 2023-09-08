// 
// Author: Mathis Cadier mathis.cadier.430@cranfield.ac.uk
//

// Computes matrix-vector product between a sparse matrix and a vector
// Sparse Matrix A is stored in 2 formats : CSR and ELLPACK
// We generate randomly a vector x 


#include <iostream>

#include "../mmio.h" // For reading matrix.mtx

#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers

// Define the size of the block of threads
#define XBD 16
#define YBD 8
const dim3 BLOCK_DIM(XBD,YBD);

// Define the number of time, we will execute our GPU kernel
const int ntimes=100;

// Device function that does reduction on the variable sdata by unrolling the last warp
__device__ void rowReduce2dBlock(volatile double *sdata, int tid, int s) {
  switch(s){
  case 16:  sdata[tid] += sdata[tid + 16];
  case  8:  sdata[tid] += sdata[tid +  8];
  case  4:  sdata[tid] += sdata[tid +  4];
  case  2:  sdata[tid] += sdata[tid +  2];
  case  1:  sdata[tid] += sdata[tid +  1];
  }
}

//////////////////////////////////////////////////
//                                              //
// -------------- CSR FUNCTIONS --------------- //
//                                              //
//////////////////////////////////////////////////

// Simple CPU implementation of matrix-vector product for CSR
void CSR_CpuMatrixVector(int rows, int cols, const int* IRP, const int* J, const double* val, const double* x, double* y) {  
  for (int i = 0; i < rows; ++i) {
    double t=0.0;
    for (int j = IRP[i]-1; j < IRP[i+1]-1; ++j) {
      t += val[j] * x[J[j]-1];
    }
    y[i] = t;
  }
}

// GPU implementation of matrix-vector product for CSR
// where 2d blocks of threads are assigned for each row of the sparse matrix
__global__ void CSR_gpuMatrixVector2dBlock(int rows, int cols, const int* IRP, const int* J, const double* val, const double* x, double* y) {
  __shared__ double ax[YBD][XBD];
  // each thread loads and multiplicates one element
  int tr     = threadIdx.y;
  int tc     = threadIdx.x;
  int i      = blockIdx.x*blockDim.y + tr;
  int s;
  ax[tr][tc] = 0.0;
  
  // Working only with threads that have an index (in the 2d block) lower than the number of rows
  if (i < rows) {
    // Calculating the matrix-vector product and storing it in the variable ax
    int j   = tc+IRP[i]-1; 
    double t  = 0.0;
    for ( ; j<IRP[i+1]-1; j += XBD) {
      t += val[j] * x[J[j]-1];
    }
    ax[tr][tc] = t;
  }
  __syncthreads();
  
  // Operating reduction on the variable ax
  // Doing reduction in shared memory
  for (s=XBD/2; s >=32; s >>=1){
    if (tc<s)
      ax[tr][tc] += ax[tr][tc+s]; 
    __syncthreads();
  }

  s = min(16,XBD/2);
  if (tc < s) rowReduce2dBlock(&(ax[tr][0]),tc,s);
  
  // Writing the result for this block to global memory
  if ((tc == 0)&&(i<rows))
    y[i] = ax[tr][tc];
  
}

////////////////////////////////////////////////////////////
//                                                        //
// ---------------- ELLPACK FUNCTIONS ------------------- //
//                                                        //
////////////////////////////////////////////////////////////

// Simple CPU implementation of matrix-vector product for ELLPACK
void ELL_CpuMatrixVector(int rows, int cols, int MAXNZ, const int* JA, const double* AS, const double* x, double* y) 
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

// GPU implementation of matrix-vector product for ELLPACK
// where 2d blocks of threads are assigned for each row of the sparse matrix
__global__ void ELL_gpuMatrixVector2dBlock(int rows, int cols, int MAXNZ, const int* JA, const double* AS, const double* x, double* y) {
  __shared__ double ax[YBD][XBD];
  // each thread loads and multiplicates one element
  int tr     = threadIdx.y;
  int tc     = threadIdx.x;
  int i      = blockIdx.x*blockDim.y + tr;
  int s;
  ax[tr][tc] = 0.0;
  
  // Working only with threads that have an index (in the 2d block) lower than the number of rows
  if (i < rows) {
    // Calculating the matrix-vector product and storing it in the variable ax
    int idx = i*MAXNZ + tc;
    int j   = tc; 
    double t  = 0.0;
    for ( ; j<MAXNZ; j += XBD) {
      t += AS[idx]*x[JA[idx]-1];
      idx += XBD;
    }
    ax[tr][tc] = t;
  }
  __syncthreads();
  
  // Operating reduction on the variable ax
  // Doing reduction in shared memory
  for (s=XBD/2; s >=32; s >>=1){
    if (tc<s)
      ax[tr][tc] += ax[tr][tc+s]; 
    __syncthreads();
  }

  s = min(16,XBD/2);
  if (tc < s) rowReduce2dBlock(&(ax[tr][0]),tc,s);
  
  // Writing the result for this block to global memory
  if ((tc == 0)&&(i<rows))
    y[i] = ax[tr][tc];
  
}

//////////////////////////////////////////////////
//                                              //
// ------------- MAIN FUNCTION ---------------- //
//                                              //
//////////////////////////////////////////////////



int main(int argc, char** argv) {
  
  // Declaring variables
  MM_typecode matcode;
  int M, N, nz;   
  int *I, *J;
  double *val;
  
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
  
  
  

  // ----------------------------------- Host memory initialisation ----------------------------------- //
  
  /**************************************/
  /* reseve memory for x and y matrices */
  /**************************************/
  
  double* h_x = (double*) malloc(M * sizeof(double));
  
  double* h_y_ell = (double*) malloc(M * sizeof(double));
  double* h_y_csr = (double*) malloc(M * sizeof(double));
  
  double* h_y_d_csr = (double*) malloc(M * sizeof(double));
  double* h_y_d_ell = (double*) malloc(M * sizeof(double));
  
  // Generating randomly x (but always with the same values thanks to srand(12345))
  srand(12345);
  for (int row = 0; row < M; ++row){
      h_x[row] = 100.0f * ((double) rand()) / RAND_MAX;
  }
  
  std::cout << "Matrix-vector product: 2D block of threads per row version "  <<std::endl;
  std::cout << "Test case: " << M  << " x " << N << " with " << nz << " NZ " << std::endl;
    
  /**********************************/
  /* reseve memory for CSR matrices */
  /**********************************/
  
  int *h_IRP = (int *) calloc((M + 1), sizeof(int));
  int *h_IRP_v = (int *) calloc((M + 1), sizeof(int));
  
  int *h_JA_csr = (int *) malloc(nz * sizeof(int));
  double *h_AS_csr = (double *) malloc(nz * sizeof(double));
  
  
  /***************************/
  /* reseve memory for MAXNZ */
  /***************************/
  
  int* NZ = (int *) calloc(M, sizeof(int));    
  int MAXNZ = 0;
  
  // Calculating MAXNZ for the ELLPACK format and IRP for the CSR format
  for (int i = 0; i < nz; i++) {
      h_IRP[I[i]+1]++;
      NZ[I[i]]++;
  }
  
  for (int i = 0; i < M; i++) {
      if (NZ[i] > MAXNZ){
          MAXNZ = NZ[i];
      }
  }
  
  fprintf(stdout, "\n");
  fprintf(stdout, "MAXNZ : %d \n", MAXNZ);
  fprintf(stdout, "\n");
  
  // Calculating JA and AS for CSR format
  h_IRP[0] = 1;
  h_IRP_v[0] = 1;
  
  for (int i = 1; i <= M; i++) {
      h_IRP[i] += h_IRP[i-1];
      h_IRP_v[i] = h_IRP[i];
  }
      
  for (int i = 0; i < nz; i++){
      int row = I[i];
      int src = h_IRP_v[row]-1;
      h_JA_csr[src] = J[i]+1;  
      h_AS_csr[src] = val[i];
      h_IRP_v[row]++;        
  }
  
  /**********************************/
  /* reseve memory for ELL matrices */
  /**********************************/
  
  int *h_JA_ell = (int *) calloc(M * MAXNZ, sizeof(int));
  double *h_AS_ell = (double *) calloc(M * MAXNZ, sizeof(double));
  
  // Calculating JA and AS for ELLPACK format
  for (int i = 0; i <= M; i++) {
      h_IRP_v[i] = 0;
  }
  
  for (int i = 0; i < nz; i++){
      int row = I[i];
      int src = h_IRP_v[row];
      h_JA_ell[row * MAXNZ + src] = J[i]+1;  
      h_AS_ell[row * MAXNZ + src] = val[i];
      h_IRP_v[row]++;        
  }
  
  
  
  
  
// ---------------------------------- Device memory initialisation -------------------------------------- //
  
  //  Allocate memory space on the device:
  // For CSR Format
  int *d_IRP, *d_JA_csr;
  double *d_AS_csr;
  
  checkCudaErrors(cudaMalloc((void**) &d_IRP, (M + 1) * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_JA_csr, nz * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_AS_csr, nz * sizeof(double)));
  
  // For ELLPACK Format
  int *d_JA_ell;
  double *d_AS_ell;
      
  checkCudaErrors(cudaMalloc((void**) &d_JA_ell, M * MAXNZ * sizeof(int)));
  checkCudaErrors(cudaMalloc((void**) &d_AS_ell, M * MAXNZ * sizeof(double)));
  
  // For x and y
  double *d_x, *d_y_csr, *d_y_ell;
  
  checkCudaErrors(cudaMalloc((void**) &d_x, M * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &d_y_csr, M * sizeof(double)));
  checkCudaErrors(cudaMalloc((void**) &d_y_ell, M * sizeof(double)));
  
  
  // Copy matrices and arrays from the host (CPU) to the device (GPU):
  // For CSR Format
  checkCudaErrors(cudaMemcpy(d_IRP, h_IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_JA_csr, h_JA_csr,  nz * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_AS_csr, h_AS_csr,  nz * sizeof(double), cudaMemcpyHostToDevice));
  
  // For ELLPACK Format
  checkCudaErrors(cudaMemcpy(d_JA_ell, h_JA_ell,  M * MAXNZ * sizeof(int), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_AS_ell, h_AS_ell,  M * MAXNZ * sizeof(double), cudaMemcpyHostToDevice));
  
  // For x and y
  checkCudaErrors(cudaMemcpy(d_x, h_x,  M * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_csr, h_y_csr,  M * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y_ell, h_y_ell,  M * sizeof(double), cudaMemcpyHostToDevice));
  
  
  


// ------------------------------------- Calculations on the CPU ---------------------------------------- //
  
  float flopcnt=2.e-6*nz;
  
  // Creating the CUDA SDK timer
  StopWatchInterface* timer = 0;
  sdkCreateTimer(&timer);
  
  // -------------------------- Calculations CSR on the CPU --------------------------- //

  timer->start();
  CSR_CpuMatrixVector(M, N, h_IRP, h_JA_csr, h_AS_csr, h_x, h_y_csr);
  timer->stop();
  
  float cpuflops=flopcnt/ timer->getTime();
  std::cout << "CSR  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;
  
  // ------------------------ Calculations ELLPACK on the CPU ------------------------- //
  
  timer->reset();
  flopcnt=2.e-6*nz;
  
  timer->start();
  ELL_CpuMatrixVector(M, N, MAXNZ, h_JA_ell, h_AS_ell, h_x, h_y_ell); 
  timer->stop();
  
  cpuflops=flopcnt/ timer->getTime();
  std::cout << "ELLPACK  CPU time: " << timer->getTime() << " ms." << " GFLOPS " << cpuflops << std::endl;
  fprintf(stdout, "\n");
  
  
  
  

// ------------------------------------- Calculations on the GPU ---------------------------------------- //

  // Calculate the dimension of the grid of blocks. A 1D grid suffices.
  const dim3 GRID_DIM((M - 1+ BLOCK_DIM.y)/ BLOCK_DIM.y  ,1);
  
  // -------------------------- Calculations CSR on the GPU --------------------------- //
  double bdwdth;
  float tmlt = 0.0;
  for (int t=0; t < ntimes; t ++ ) {
      timer->reset();
      
      timer->start();
      CSR_gpuMatrixVector2dBlock<<<GRID_DIM, BLOCK_DIM >>>(M, N, d_IRP, d_JA_csr, d_AS_csr, d_x, d_y_csr);
      checkCudaErrors(cudaDeviceSynchronize());
      
      timer->stop();
      tmlt += timer->getTime();
  }
  
  // Average runtime
  tmlt /= ntimes;  
  float gpuflops=flopcnt/ tmlt;
  bdwdth = ((double)(2*nz-M)*sizeof(double))/tmlt;
  bdwdth *= 1.e-6;
  
  std::cout << "CSR 2d block GPU time: " << tmlt << " ms." << " GFLOPS " << gpuflops<<std::endl;
  std::cout << "Measured bandwidth: " << bdwdth << " GB/s" << std::endl;
  fprintf(stdout, "\n");

  // ------------------------ Calculations ELLPACK on the GPU ------------------------- //
  
  tmlt = 0.0;
  for (int t=0; t < ntimes; t ++ ) {
      timer->reset();
      
      timer->start();
      ELL_gpuMatrixVector2dBlock<<<GRID_DIM, BLOCK_DIM >>>(M, N, MAXNZ, d_JA_ell, d_AS_ell, d_x, d_y_ell);
      checkCudaErrors(cudaDeviceSynchronize());
      
      timer->stop();
      tmlt += timer->getTime();
  }
  
  // Average runtime
  tmlt /= ntimes;    
  gpuflops=flopcnt/ tmlt;
  bdwdth = ((double)(2*nz-M)*sizeof(double))/tmlt;
  bdwdth *= 1.e-6;
  
  std::cout << "ELLPACK 2d block GPU time: " << tmlt << " ms." << " GFLOPS " << gpuflops<<std::endl;
  std::cout << "Measured bandwidth: " << bdwdth << " GB/s" << std::endl;
  fprintf(stdout, "\n");
  

  
  
  
  
// -------------------------------------------------- Results -------------------------------------------------- //
  

  // Download the resulting vector d_y from the device and store it in h_y_d
  // For CSR Format
  checkCudaErrors(cudaMemcpy(h_y_d_csr, d_y_csr, M * sizeof(double),cudaMemcpyDeviceToHost));
  
  // For ELLPACK Format
  checkCudaErrors(cudaMemcpy(h_y_d_ell, d_y_ell, M * sizeof(double),cudaMemcpyDeviceToHost));
  
  // Now let's check if the results are the same for CPU CSR and GPU CSR
  double reldiff_csr = 0.0f;
  double diff_csr = 0.0f;
  
  for (int row = 0; row < M; ++row) {
    double maxabs_csr = std::max(std::abs(h_y_csr[row]),std::abs(h_y_d_csr[row]));
    if (maxabs_csr == 0.0) maxabs_csr=1.0;
    reldiff_csr = std::max(reldiff_csr, std::abs(h_y_csr[row] - h_y_d_csr[row])/maxabs_csr);
    diff_csr = std::max(diff_csr, std::abs(h_y_csr[row] - h_y_d_csr[row]));
  }
  std::cout << "CSR 2d block : Max diff = " << diff_csr << "  Max rel diff = " << reldiff_csr << std::endl;
  
  
  // Now let's check if the results are the same for CPU ELLPACK and GPU ELLPACK
  double reldiff_ell = 0.0f;
  double diff_ell = 0.0f;
  
  for (int row = 0; row < M; ++row) {
    double maxabs_ell = std::max(std::abs(h_y_ell[row]),std::abs(h_y_d_ell[row]));
    if (maxabs_ell == 0.0) maxabs_ell=1.0;
    reldiff_ell = std::max(reldiff_ell, std::abs(h_y_ell[row] - h_y_d_ell[row])/maxabs_ell);
    diff_ell = std::max(diff_ell, std::abs(h_y_ell[row] - h_y_d_ell[row]));
  }
  std::cout << "ELLPACK 2d block : Max diff = " << diff_ell << "  Max rel diff = " << reldiff_ell << std::endl;
  fprintf(stdout, "\n");
  
  
  
  

// ------------------------------------------------ Cleaning up ------------------------------------------------ //

  delete timer;

  checkCudaErrors(cudaFree(d_IRP));
  checkCudaErrors(cudaFree(d_JA_csr));
  checkCudaErrors(cudaFree(d_JA_ell));
  checkCudaErrors(cudaFree(d_AS_csr));
  checkCudaErrors(cudaFree(d_AS_ell));
  checkCudaErrors(cudaFree(d_x));
  checkCudaErrors(cudaFree(d_y_ell));
  checkCudaErrors(cudaFree(d_y_csr));

  delete[] h_IRP;
  delete[] h_IRP_v;
  delete[] h_JA_csr;
  delete[] h_JA_ell;
  delete[] h_AS_csr;
  delete[] h_AS_ell;
  delete[] h_x;
  delete[] h_y_csr;
  delete[] h_y_ell;
  delete[] h_y_d_csr;
  delete[] h_y_d_ell;
  
  return 0;
}
