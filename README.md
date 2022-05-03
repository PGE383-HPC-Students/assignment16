# Homework Assignment 16

The weak form of the steady-state pressure diffusivity equation is

\begin{align}
0 &= \int_{\partial \Omega} \lambda(\vec{x}) \delta p(\vec{x}) \nabla p(\vec{x}) \textrm{d}S - \int_{\Omega}  \lambda(\vec{x}) \nabla \delta p(\vec{x}) 
\cdot \nabla p(\vec{x})\textrm{d}V \\
  &= \int_{\partial \Omega} \delta p(\vec{x}) \vec{v} \textrm{d}S - \int_{\Omega}  \lambda(\vec{x}) \nabla \delta p(\vec{x}) 
\cdot \nabla p(\vec{x})\textrm{d}V \\
 &= l(\delta p) - a(p, \delta p)
\end{align}

where $\lambda$ is a mobility function, $\delta p$ is a "test function" and $p$ is a "trial function" and unknown pressure field variable.  Your assignment is to use [Gripap.jl](https://gridap.github.io/Gridap.jl/stable/) to solve this equation for a one-dimensional discretization.  Specifically, you should
complete the function `fe_solver(number_of_elements::Integer, λ::Function, left_bc::Real, right_bc::Real)` in [assignmetn16.jl](src/assignment16.jl). `number_of_elements` are the number of finite elements to use in the discretization, `λ` is the mobility (defined as a function), `left_bc` and `right_bc` are the constant left and right Dirchelet boundary conditions respectively.  There are no Nuemann boundary conditions, i.e. $\vec{v} = \lambda \nabla p = 0$.  The function should return the `Gridap` solver object (i.e. the thing returned by the `solve` function).  

You should use uniformly defined elements on the one-dimensional domain $(0, 1)$, the reference element should use linear Lagrange interpolation functions, the quadrature should use 2-point Gauss integration.


## Testing

To see if your answer is correct, run the following command at the Terminal
command line from the repository's root directory

```bash
julia --project=. -e "using Pkg; Pkg.test()"
```

the tests will run and report if passing or failing.
