## **1. Matrix Row/Column Sums**

Your first task is to create a simple matrix row and column sum application in CUDA. The code skeleton is already given to you in *matrix_sums.cu*. Edit that file, paying attention to the FIXME locations, so that the output when run is like this:

```
row sums correct!
column sums correct!
```

If you have trouble, you can look at *matrix_sums_solution.cu* for a complete example.

## **2. Profiling**

We'll introduce something new: the profiler (in this case, Nsight Compute). We'll use the profiler first to time the kernel execution times, and then to gather some "metric" information that will possibly shed light on our observations.

It's necessary to complete task 1 first.

Then, launch Nsight as follows:
(you may want to make your terminal session wide enough to make the output easy to read)

```
lsfrun nv-nsight-cu-cli ./matrix_sums
```

What does the output tell you?
Can you locate the lines that identify the kernel durations?
Are the kernel durations the same or different?
Would you expect them to be the same or different?


Next, launch *Nsight* as follows:

```
lsfrun nv-nsight-cu-cli --metrics l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum ./matrix_sums
```

Our goal is to measure the global memory load efficiency of our kernels. In this case we have asked for two metrics: "*l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum*" (the number of global memory load requests) and "*l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum*" (the number of sectors requested for global loads). This first metric above represents the denominator (requests) of the desired measurement (transactions per request) and the second metric represents the numerator (transactions). Dividing these numbers will give us the number of transactions per request. 

What similarities or differences do you notice between the *row_sum* and *column_sum* kernels?
Do the kernels (*row_sum*, *column_sum*) have the same or different efficiencies?
Why?
How does this correspond to the observed kernel execution times for the first profiling run?

Can we improve this?  (Stay tuned for the next CUDA training session.)

Here is a useful blog to help you get familiar with Nsight Compute: https://devblogs.nvidia.com/using-nsight-compute-to-inspect-your-kernels/

