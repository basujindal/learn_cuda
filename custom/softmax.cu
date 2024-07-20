#include <stdio.h>

// error checking macro
#define cudaCheckErrors(msg) \
    do { \
        cudaError_t __err = cudaGetLastError(); \
        if (__err != cudaSuccess) { \
            fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
                msg, cudaGetErrorString(__err), \
                __FILE__, __LINE__); \
            fprintf(stderr, "*** FAILED - ABORTING\n"); \
            exit(1); \
        } \
    } while (0)


const size_t DSIZE = 16384;      // matrix side dimension
// const size_t DSIZE = 1024;      // matrix side dimension
const int block_size = 256;  // CUDA maximum is 1024
// matrix row-sum kernel


__global__ void row_sums(const float *A, float *sums, size_t ds){

  int idx = threadIdx.x;
  __shared__ float sdata[block_size];
  sdata[idx] = 0.0f;

  for(int i = 0; i < ds/blockDim.x; i++) sdata[idx] += A[ds*blockIdx.x + i*blockDim.x + idx];
  
  for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s) sdata[idx] += sdata[idx + s];
  }
  
  if (idx == 0) sums[blockIdx.x] = sdata[0];

}


__global__ void softmax(float *A, float *sums, size_t ds){

  int idx = threadIdx.x;
  __shared__ float sdata[block_size];
  sdata[idx] = 0.0f;

  for(int i = 0; i < ds/blockDim.x; i++){
    float val = expf(A[ds*blockIdx.x + i*blockDim.x + idx]);
    A[ds*blockIdx.x + i*blockDim.x + idx] = val;
    sdata[idx] += val;
  }
  
  for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s) sdata[idx] += sdata[idx + s];
  }
  
  if (idx == 0) sums[blockIdx.x] = sdata[0];
  
  for(int i = 0; i < ds/blockDim.x; i++) A[ds*blockIdx.x + i*blockDim.x + idx] /= sdata[0];
  

}


bool validate(float *data, size_t sz){
  
  for (size_t i = 0; i < sz; i++){
    // printf("%f\n", expf(0.005)*(float)sz);
    float val = expf(0.005)*(float)sz;
    if (data[i] - val > 0.001) {printf("results mismatch at %lu, was: %f, should be: %f\n", i, data[i], val); return false;}
  }
    return true;
}
int main(){

  float *h_A, *h_sums, *d_A, *d_sums;
  h_A = new float[DSIZE*DSIZE];  // allocate space for data in host memory
  h_sums = new float[DSIZE]();
  for (int i = 0; i < DSIZE*DSIZE; i++)  // initialize matrix in host memory
    h_A[i] = 0.005f;
  cudaMalloc(&d_A, DSIZE*DSIZE*sizeof(float));  // allocate device space for A
  cudaMalloc(&d_sums, DSIZE*sizeof(float));  // allocate device space for vector d_sums
  cudaCheckErrors("cudaMalloc failure"); // error checking
  // copy matrix A to device:
  cudaMemcpy(d_A, h_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");
  //cuda processing sequence step 1 is complete
  //row_sums<<<DSIZE, block_size>>>(d_A, d_sums, DSIZE);
  softmax<<<DSIZE, block_size>>>(d_A, d_sums, DSIZE);
  cudaCheckErrors("kernel launch failure");
  //cuda processing sequence step 2 is complete
  // copy vector sums from device to host:
  cudaMemcpy(h_sums, d_sums, DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  //cuda processing sequence step 3 is complete
  cudaCheckErrors("1 kernel execution failure or cudaMemcpy H2D failure");
  if (!validate(h_sums, DSIZE)) return -1; 
  printf("row sums correct!\n");

  cudaMemcpy(h_A, d_A, DSIZE*DSIZE*sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy D2H failure");

  for(int i = 0; i < 100; i++){
    // printf("%f\n %f\n", h_A[i], 1/(float)DSIZE);
    if(h_A[i] != 1/(float)DSIZE
    ) {printf("results mismatch at %d, was: %f, should be: %f\n", i, h_A[i], 1/float(DSIZE)); return -1;}
  }
    printf("softmax correct!\n");


  return 0;
}
  
