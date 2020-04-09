# Homework Assignment 16

Gauss quadrature (i.e. `fixed_quad()`) in `scipy` is a clever procedure for numerically evaluating definite integrals. Gauss quadrature gets more accurate as the number (order) of quadrature points increases.

Your assignment is to create a parallel scheme with `mpi4py`, using only `send()`, `isend()`, `recv()`, and `gather()` calls, that can be used to generate a table of the number of Gauss points and the corresponding value of the integral from 1 to 50.

You should **not** use `scatter()` to evenly distribute an array of the integration orders, because the higher orders are more computationally expensive and this method will cause most of the work to be done on the highest numbered rank, leaving the others unutilized; but, instead, try to design a program that will keep all ranks busy computing the integrals until they are all complete.  One idea is to use a boss/worker model, where the rank 0 processor just serves to hand out work to other processors when they are not busy.

Complete the function `generate_table()` in [assignment16.py](assignment16.py).  Each rank that is assigned work should store a list of `[integration order, integral value]` pairs in the class attribute `my_table` that is then gathered and returned.

The class constructor takes a function defining the integral, the integration limits, and an MPI communicator as arguments.  See the tests if you need assistance in using the class.

## Testing

If you would like to check to see if your solution is correct, run the following commands at the Terminal command line:

```bash
mpiexec -np 2 python test.py
mpiexec -np 4 python test.py
```
