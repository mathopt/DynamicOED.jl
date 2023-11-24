# Theoretical Background

!!! note 
    This section serves as a short introduction. For a more detailed introduction, please refer to [this paper](https://doi.org/10.1137/110835098).

DynamicOED.jl is focused on deriving optimal designs for differential (algebraic) systems of the form

```math 
\begin{aligned}
0 &= f(\dot{x}, x, p, t, u) \\ 
y &= h(x, p, t, u)
\end{aligned}
```
