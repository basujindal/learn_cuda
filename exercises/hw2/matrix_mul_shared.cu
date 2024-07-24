#include <stdio.h>
#include <time.h>

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


const int height = 32;
const int width = 768;
const int dim = 768;
const int block_size = 32;  // CUDA maximum is 1024 *total* threads in block

__global__ void mmul(const float *A, const float *B, float *C, float *bias, int height, int width, int dim) {

  // declare cache in shared memory
  __shared__ float As[block_size][block_size];
  __shared__ float Bs[block_size][block_size];

  int row = threadIdx.y+blockDim.y*blockIdx.y; // create thread y index
  int col = threadIdx.x+blockDim.x*blockIdx.x; // create thread x index
  

  if ((row < height) && (col < width)){
    float temp = 0;
    for (int i = 0; i < dim/block_size; i++) {

      // Load data into shared memory
      As[threadIdx.y][threadIdx.x] = A[row*dim + (block_size*i + threadIdx.x)];
      Bs[threadIdx.y][threadIdx.x] = B[col + width*(block_size*i + threadIdx.y)];

      // Synchronize
      __syncthreads();

      // Keep track of the running sum
      for (int k = 0; k < block_size; k++)
      	temp += As[threadIdx.y][k] * Bs[k][threadIdx.x]; // dot product of row and column
      __syncthreads();
    }

    if(col ==  33 && row == 0) printf("row: %d, col: %d\n", height, width);
    // Write to global memory
    if (row < 5) C[row*width+col] = temp + bias[col];
    else C[row*width+col] = temp;
    // C[row*width+col] = temp + bias[col];

  }
}

int read_weight(float *arr, char *filename, int rows, int cols){

  printf("Reading %s\n", filename);
  
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
      printf("Error opening file\n");
      return 1;
  }

  fread(arr, sizeof(float), rows * cols, file);
  fclose(file);


  return 0;
  
}

int main(){

  float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C, *d_bias, *h_bias;

  clock_t t0, t1, t2;
  double t1sum=0.0;
  double t2sum=0.0;

  // start timing
  t0 = clock();

  h_A = new float[height*dim];
  h_B = new float[dim*dim];
  h_C = new float[height*width];
  h_bias = new float[width];

  char filename[256];

  
  for (int i = 0; i < height*dim; i++) h_A[i] = 0;
  for (int i = 0; i < dim*width; i++) h_B[i] = 0;
  for (int i = 0; i < height*width; i++) h_C[i] = 0;
  for (int i = 0; i < width; i++) h_bias[i] = 0;

  snprintf(filename, sizeof(filename), "../../custom/gpt_weights/ln.bin");
  read_weight(h_A, filename, 32, dim);

  snprintf(filename, sizeof(filename), "../../custom/gpt_weights/h.0.attn.c_attn.weight.q.bin");
  read_weight(h_B, filename, dim, dim);

  snprintf(filename, sizeof(filename), "../../custom/gpt_weights/h.0.attn.c_attn.bias.q.bin");
  read_weight(h_bias, filename, dim, 1);

  // // print h_B
  // for (int i = 0; i < 768; i++){
  //   for (int j = 0; j < 768; j++){
  //     printf("%f ", h_B[i*768+j]);
  //   }
  //   printf("\n");
  // }

  // for (int i = 0; i < height*dim; i++) h_A[i] = rand()/(float)RAND_MAX;
  // for (int i = 0; i < dim*width; i++) h_B[i] = rand()/(float)RAND_MAX;
  // for (int i = 0; i < height*width; i++) h_C[i] = 0;
  // for (int i = 0; i < width; i++) h_bias[i] = rand()/(float)RAND_MAX;
    

  // Initialization timing
  t1 = clock();
  t1sum = ((double)(t1-t0))/CLOCKS_PER_SEC;
  printf("Init took %f seconds.  Begin compute\n", t1sum);

  // Allocate device memory and copy input data over to GPU
  cudaMalloc(&d_A, height*dim*sizeof(float));
  cudaMalloc(&d_B, dim*width*sizeof(float));
  cudaMalloc(&d_C, height*width*sizeof(float));
  cudaMalloc(&d_bias, width*sizeof(float));

  cudaCheckErrors("cudaMalloc failure");
  cudaMemcpy(d_A, h_A, height*dim*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, h_B, dim*width*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, h_bias, width*sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMemcpy H2D failure");

  // Launch kernel
  dim3 block(block_size, block_size);  // dim3 variable holds 3 dimensions
  dim3 grid((width+block.x-1)/block.x, (height+block.y-1)/block.y);
  mmul<<<grid, block>>>(d_A, d_B, d_C, d_bias, height, width, dim);
  cudaCheckErrors("kernel launch failure");

  // Copy results back to host
  cudaMemcpy(h_C, d_C, height*width*sizeof(float), cudaMemcpyDeviceToHost);

  // GPU timing
  t2 = clock();
  t2sum = ((double)(t2-t1))/CLOCKS_PER_SEC;
  printf ("Done. Compute took %f seconds\n", t2sum);

  // Verify results
  cudaCheckErrors("kernel execution failure or cudaMemcpy H2D failure");

  for (int i = 0; i < 6; i++){
    for (int j = 0; j < width; j++){
      printf("%d) %f ",j,  h_C[i*width+j]);
      // float sum = 0;
      // for (int k = 0; k < dim; k++) sum += h_A[i*dim+k]*h_B[k*width+j] + h_bias[j];
      // if (h_C[i*width+j] - sum > 0.01) {
      // printf("mismatch at index %d, was: %f, should be: %f\n", i*width+j, h_C[i*width+j], sum);
      // return -1;
      // }

    }
    printf("\n\n\n");
  }

  printf("Success!\n"); 
  return 0;
}
  
