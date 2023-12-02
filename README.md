[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# levy-SSMs

The implementation of the sequential MCMC methodology introduced in [__Generalised Hyperbolic State-space Models for Inference in Dynamic Systems__](https://arxiv.org/abs/2309.11422) by Yaman Kındap and Simon Godsill and simulation algorithms for the underlying Generalised inverse-Gaussian (GIG) subordinator are provided.

The implementation can handle the more general case of normal variance-mean mixture Lévy processes which include the normal-gamma process and normal tempered stable process. Simulation algorithms that utilise adaptive truncation of the series representations and approximation of the residual as a Gaussian discussed in [__Point process simulation of generalised hyperbolic Lévy processes__](https://link.springer.com/article/10.1007/s11222-023-10344-x) by Kındap and Godsill for the gamma and tempered stable processes are additionally provided.