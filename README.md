# Simple LLM in pure CUDA C

## To Do

- [x] Softmax
- [] Attention
- [] Linear Layer
- [] RMSNorm
- [] GELU



## Learning Material

The materials in this repository accompany the CUDA Training Series presented at ORNL and NERSC.

You can find the slides and presentation recordings at https://www.olcf.ornl.gov/cuda-training-series/


## **1. Compile Code*

nvcc -arch=sm_75 --allow-unsupported-compiler vector_add.cu -o test && ./test

## **2. Profiling Experiments**

nvcc -arch=sm_75 --allow-unsupported-compiler vector_add.cu -o test && sudo /usr/local/cuda/bin/ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./test

nvcc -arch=sm_75 --allow-unsupported-compiler matrix_sums.cu -o test && sudo /usr/local/cuda/bin/ncu --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum --section SpeedOfLight --section MemoryWorkloadAnalysis ./test

