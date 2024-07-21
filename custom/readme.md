## Softmax

nvcc -arch=sm_75 --allow-unsupported-compiler softmax.cu -o test && sudo /usr/local/cuda/bin/ncu --section SpeedOfLight --section MemoryWorkloadAnalysis ./test

### Profiling Results

- DSIZE = 16384
- BSIZE = 1024
- Duration = 13.42 ms

## To Do

- [ ] Make it stable by substracting the maximum value from the input array