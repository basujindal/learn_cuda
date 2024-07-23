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


const int N = 6;      // matrix side dimension
const int Dim = 1024;
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block  
const int block_size_softmax = 1024;  // CUDA maximum is 1024 *total* threads in block


__global__ void softmax(float *QK, size_t ds){

  int idx = threadIdx.x;
  __shared__ float sdata[block_size_softmax];
  sdata[idx] = 0.0f;
  float val = 0;

  __shared__ float max_val;
  max_val = 0.0f;

  // Find the maximum value in the block

  for (int index = start_index + idx; index < end_index; index += blockDim.x) {
    if (index < ds*ds) sdata[idx] = max(A[index], sdata[idx]);
  }

  for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s) sdata[idx] = max(sdata[idx], sdata[idx + s]);
  }
  __syncthreads();

  if (idx == 0) max_val = sdata[0];
  __syncthreads();

  sdata[idx] = 0.0f;

  for(int i = 0; i < ds/blockDim.x; i++){
    val = expf(QK[ds*blockIdx.x + i*blockDim.x + idx]);
    QK[ds*blockIdx.x + i*blockDim.x + idx] = val;
    sdata[idx] += val;
  }
  
  for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s) sdata[idx] += sdata[idx + s];
  }
  __syncthreads();
  
  for(int i = 0; i < ds/blockDim.x; i++) QK[ds*blockIdx.x + i*blockDim.x + idx] /= sdata[0];

  if(idx == 0) printf("sdata[0]: %f\n", sdata[0]);
  
}

__global__ void softmax_max(float *A, size_t ds){

  int idx = threadIdx.x;
  __shared__ float sdata[block_size_softmax];
  sdata[idx] = 0.0f;
  float val = 0.0f;
  float max_val = 0.0f;

  for(int i = 0; i < ds/blockDim.x; i++){
    sdata[idx] = max(A[ds*blockIdx.x + i*blockDim.x + idx], sdata[idx]);
  }

  for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s) sdata[idx] = max(sdata[idx], sdata[idx + s]);
  }
  __syncthreads();

  if (idx == 0) max_val = sdata[0];
  sdata[idx] = 0.0f;

  __syncthreads();

  for(int i = 0; i < ds/blockDim.x; i++){
    val = expf(A[ds*blockIdx.x + i*blockDim.x + idx] - max_val);
    A[ds*blockIdx.x + i*blockDim.x + idx] = val;
    sdata[idx] += val;
  }

  for(int s = blockDim.x/2; s > 0; s/=2){
    __syncthreads();
    if (idx < s) sdata[idx] += sdata[idx + s];
  }
  __syncthreads();
  
  for(int i = 0; i < ds/blockDim.x; i++) A[ds*blockIdx.x + i*blockDim.x + idx] /= sdata[0];

  if (idx == 0) printf("A: %f\n", A[ds*blockIdx.x]);
  

}


__global__ void matmul(const float *Attn, const float *V, float *C, int Dim, int N) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];
  
  int col = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int row = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  if ((row < N) && (col < Dim)){
    float temp = 0;

    for (int i = 0; i < N/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = Attn[row*N + (block_size*i + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = V[col + Dim*(block_size*i + threadIdx.y)];

      __syncthreads();

      for (int k = 0; k < block_size; k++) temp +=  As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column

      __syncthreads();

    }
    // Write to global memory
    C[row*Dim+col] = temp;
  }
}


__global__ void QK_V(const float *QK, const float *V, float *C, int Dim, int N) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];
  
  int col = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  int row = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index

  if ((row < N) && (col < Dim)){
    float temp = 0, val, sum = 0;

    for (int i = 0; i < N/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = expf(QK[row*N + (block_size*i + threadIdx.x)]);
      Bs[threadIdx.y][threadIdx.x] = V[col + Dim*(block_size*i + threadIdx.y)];

      __syncthreads();

      for (int k = 0; k < block_size; k++){
        val = As[threadIdx.y][k];
      	temp +=  val * Bs[k][threadIdx.x]; // dot product of row and column
        sum+=val;
      }

      __syncthreads();

    }

    // Write to global memory
    C[row*Dim+col] = temp/sum;
  }
}


int validateQK_V(float *h_QK, float *h_V, float *h_ACT, int N, int Dim){

  float sums[N];

  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      h_QK[i*N+j] = expf(h_QK[i*N+j]);
      sums[i] += h_QK[i*N+j];
    }
  }


  for(int i = 0; i < N; i++) for (int j = 0; j < N; j++) h_QK[i*N+j] /= sums[i];

  for (int i = 0; i < N; i++){
    for (int j = 0; j < Dim; j++){
      float temp = 0;
      for (int k = 0; k < N; k++) temp += h_QK[i*N+k]*h_V[k*Dim+j];

      if (temp - h_ACT[i*Dim+j] > 0.1) {
        printf("results mismatch at %d, was: %f, should be: %f\n", i*Dim+j, h_ACT[i*Dim+j], temp);
        return -1;
      }
    }
  }

  printf("softmax correct!\n");
  return 0;
}

int validateSoftmax(float *h_QK, float *h_sout, int N){

  float sums[N];
  for (int i = 0; i < N; i++) sums[i] = 0;

  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      h_QK[i*N+j] = expf(h_QK[i*N+j]);
      sums[i] += h_QK[i*N+j];
    }
  }


  for(int i = 0; i < N; i++) for (int j = 0; j < N; j++) h_QK[i*N+j] /= sums[i];


  for (int i = 0; i < N; i++){
    for (int j = 0; j < N; j++){
      if (h_QK[i*N+j] - h_sout[i*N+j] > 0.001) {
        printf("results mismatch at %d, was: %f, should be: %f\n", i*N+j, h_sout[i*N+j], h_QK[i*N+j]);
        return -1;
      }
    }
  }

  printf("softmax correct!\n");
  return 0;
}

int main(){

    float *h_QK, *h_V, *h_ACT, *h_sout;
    float *d_QK, *d_V, *d_ACT;

    h_QK = new float[N*N];
    h_V = new float[N*Dim];
    h_ACT = new float[N*Dim];
    h_sout = new float[N*N];

    for (int i = 0; i < N*N; i++) h_QK[i] = rand()/(float)RAND_MAX;
    for (int i = 0; i < N*Dim; i++) h_V[i] = rand()/(float)RAND_MAX;


    cudaMalloc(&d_QK, N*N*sizeof(float));
    cudaMalloc(&d_V, N*Dim*sizeof(float));  
    cudaMalloc(&d_ACT, N*Dim*sizeof(float)); 


    cudaCheckErrors("cudaMalloc failure"); // error checking
    cudaMemcpy(d_QK, h_QK, N*N*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    cudaCheckErrors("cudaMalloc failure"); // error checking
    cudaMemcpy(d_V, h_V, N*Dim*sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckErrors("cudaMemcpy H2D failure");

    
    dim3 block(block_size, block_size);
    dim3 grid((Dim+block.x-1)/block.x, (Dim+block.y-1)/block.y);

    // Fused QK_V
    // QK_V<<<grid, block>>>(d_QK, d_V, d_ACT, Dim, N);

    // // Softmax + QK_V
    
    softmax<<<N, block_size_softmax>>>(d_QK, Dim);
    cudaCheckErrors("kernel launch failure");

    cudaMemcpy(h_sout, d_QK, N*N*sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckErrors("cudaMemcpy D2H failure");

    validateSoftmax(h_QK, h_sout, N);

    // matmul<<<grid, block>>>(d_QK, d_V, d_ACT, Dim, N);
    // cudaCheckErrors("kernel launch failure");

    // cudaMemcpy(h_ACT, d_ACT, N*Dim*sizeof(float), cudaMemcpyDeviceToHost);
    // cudaCheckErrors("cudaMemcpy D2H failure");

    // // Validate softmax(QK)*V
    // validateQK_V(h_QK, h_V, h_ACT, N, Dim);

    return 0;
}
  
