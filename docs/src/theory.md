# Theoretical Background

!!! note 
    This section serves as a short introduction. For a more detailed introduction, please refer to [this paper](https://doi.org/10.1137/110835098).

DynamicOED.jl is focused on deriving optimal experimental designs for differential (algebraic) systems of the form

```math 
\begin{aligned}
0 &= f(\dot{x}, x, p, t, u), \\ 
y &= h(x),
\end{aligned}
```

on a fixed time horizon $\mathcal T := [t_0, t_f]$ where we have states $x(\cdot) : \mathcal T \mapsto \mathbb R^{n_x}$ and $y$ and $h: \mathbb{R}^{n_x} \mapsto \mathbb{R}^{n_h}$ indicate the observed states and the measurement function, respectively.

In the optimal experimental design problem the experimental setting, e.g., initial conditions and controls, as well as the decision when to measure are determined such that the collected data from this experiment can be used to accurately estimate the model's parameters. The problem we are solving reads

```math
\begin{aligned}
    &\underset{x, G, F, w, x_0, u, \tau}{\text{min}} && \phi(F(t_f) + \tau I) \\
    &\text{subject to}
    &&  0           & =    & f(\dot x, x, p, t, u),\\
    &&& 0           & =    & f_{\dot x}(\dot x, x, p, t, u) \dot{G} + f_x(\dot x, x, p, t, u)G + f_p(\dot x, x, p, t, u),\\
    &&& \dot F   & =    & \sum_{i=1}^{n_h} w_i(t) (h^i_x(x) G)^\top (h^i_x(x) G), \\
    &&& \dot z  & =    & w,\\
    &&& x(t_0)        & =    & x_0, \quad  G(t_0) = 0, \quad F(t_0) = 0, \quad z(t_0) = 0,\\
    &&& u         & \in  & \mathcal{U},\\
    &&& w        & \in  & \mathcal W,\\
    &&& z(t_f) - M  & \leq & 0, 
\end{aligned}
```

where the first and second constraint denote the dynamical system and the variational differential (algebraic) equation for the sensitivities of the solution $x(\cdot)$ with respect to the uncertain parameters, respectively. They are given in an implicit form. Here, $f_{\dot x}$ ($f_x$) denotes the partial derivative of $f$ with respect to $\dot x$ ($x$). 

The objective $\phi(F(t_f) + \tau I)$ of Bolza type is a suited objective function, e.g., the A-criterion $\phi(F) = \textrm{trace}(F^{-1}(t_f))$, and $\tau \in \mathbb R_+$ denotes a regularization term. 
The evolution of the symmetric FIM $F : \mathcal{T} \mapsto \mathbb{R}^{n_p \times n_p}$ is influenced by the measurement function $h$, the sensitivities 
$G : \mathcal{T} \mapsto \mathbb{R}^{n_x \times n_p}$ and the sampling decisions $w(t) \in \{0,1\}^{n_h}$. The latter are the main optimization variables and represent the decision whether to measure at a given time point or not. The sampling decisions are then accumulated in the variables $z$ and constrained by $M \in \mathbb{R}^{n_h}_{+}$. Additionally, controls $u \in \mathcal{U}$ may be present to stimulate the system.
