# Fokker-Planck equations
Fokker-Planck (FP) equations, also known as _Kolmogorov forward equations_, are a central subject in the study of diffusion processes. 
FP equations describe the probability density function of a given observable (position, velocoty, etc) of a particle given its evolution in time.

In this repo, the numerical solution of the FP equation in large times is calculated.

## Particular case: FP in large times.

Given a set of random variables _Xi_, we can generate a sistem of _i_ differential equations. Since each individual particle executes a motion
which is independent of the motions of all other particles, it is possible to paralelize numerical calculations by using one thread per equation of motion. 

An stochastic dynamics can be written as an FP equation in the form

∂tP(x, t) = T ∂x^2 P(x, t) + ∂x[∂xV (x)P(x, t)],

with stationary solutions in large times

P(x, t → ∞) ∝ exp(−V (x)/T).
