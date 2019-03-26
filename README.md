## 528 Parallel code submission

- Choice of portion of code to parallelize

I chose to parallelize the loop that calculates the GP likelihood of a signal after removing the best-fit, circular Keplerian orbit. This is because, once you start evaluating the likelihood at more than ~64 periods, it dominates the serial runtime of the code.

- Choice of approach for parallelizing code

I chose to parallelize with parallel mapping, shared arrays with distributed for loops, and mapping the function onto distributed arrays. I looked at all of these methods because it was not clear to me which would be the best. I only ended up testing pmapping and distributed arrays because those approaches seemed the most flexible.

- Code performs proposed tasks

- Unit/regression tests comparing serial & parallel versions

See GP-Research/julia/test/parallel_rv_test.jl

- Code passes tests

- General code efficiency

- Implementation/optimization of multi-core parallelization

- Significant performance improvement

See figures in GP-Research/julia/figs. optimize_period_scaling_calc_time.png shows the times spent in the actual likelihood calculations while optimize_period_scaling_total_time.png shows the total runtimes. It looks like pmapping and distributed arrays have similar scaling, but pmap tends to have less overhead in its setup. This problem also seems to have strong parallel scaling.
