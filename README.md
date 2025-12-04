# Tree AR-HMM for Cell Lineages

This repository implements a **Tree Autoregressive Hidden Markov Model (Tree AR-HMM)** in **Dynamax**. This was inspired by the problem of modeling cell lineages, where each biological cell follows a Markovian latent state sequence with AR(1) Gaussian emissions, and division events create a branching tree structure. It accommodates spontaneous cell birth (e.g. coming into frame at time $t$) as well as cell death. 

In summary:
- Each cell (or agent, more generally) has a **latent discrete state** that evolves over time.  
- Emissions are **shared AR(1) dynamics** conditioned on the latent state.  
- When a cell divides, its daughters’ initial latent states are drawn from a **division-specific transition kernel** that depends on the parent’s latent state at division.

<p align="center">
  <img src="figs/fig1.png" alt="Tree AR-HMM probabilistic model" width="100%">
</p>

---

## Probabilistic Model

We index cells by $i$ (e.g. root $r$, daughters $n,m,\dots$) and time by $t$.  
Each cell has a lifetime interval over which it exists and can persist or divide.

### Latent States for Root Cells

For each root cell $r$, alive from $t = 1$ until it divides at time $\tilde t$:

- **Initial state**

  $$z_{r,1} \sim \mathrm{Cat}(\pi_0)$$

- **Within-cell transitions**

  $$z_{r,t+1} \mid z_{r,t} = k \sim \mathrm{Cat}(\pi_k), \quad t = 1,\dots,\tilde t - 1$$

Here, $\{\pi_k\}$ define the standard Markov transition matrix over discrete states.

### Emissions (Shared AR(1) Model)

For any cell $i$ and any time $t$ during its lifetime, we observe a vector $x_{i,t}$.  
Conditional on the latent state, we use a shared AR(1) emission model:

$$p\bigl(x_{i,t} \mid z_{i,t} = k,\; x_{i,t-1}\bigr) = \ell_t^{(i)}(k)$$

For example, a Gaussian AR(1):

$$x_{i,t} \mid z_{i,t} = k,\; x_{i,t-1} \sim \mathcal{N}(A_k x_{i,t-1} + b_k,\; \Sigma_k)$$

All emission parameters $\{A_k, b_k, \Sigma_k\}$ are **shared across all cells** (parents and daughters).

### Division Events and Daughters’ Initial States

Suppose a parent cell $r$ divides at time $\tilde t$, producing daughters $n$ and $m$.  
Each daughter’s initial latent state at time $\tilde t + 1$ depends on the parent’s latent state at division:

$$z_{i,\tilde t + 1} \mid z_{r,\tilde t} = k \sim \mathrm{Cat}(\tilde \pi_k), \quad i \in \{n,m\}$$

The collection $\{\tilde \pi_k\}$ defines a **division transition kernel** that maps the parent’s state at division to the daughters’ starting states.

After birth, each daughter follows the same within-cell transition dynamics:

$$z_{i,t+1} \mid z_{i,t} = k \sim \mathrm{Cat}(\pi_k), \quad t \ge \tilde t+1$$

again with AR(1) emissions as above.

### Joint Likelihood (Sketch)

Let $R$ be the number of roots, and for each root $r$, let $C_{r,t}$ be the set of cells alive at time $t$ in that lineage. For each parent–child pair $(n, c_n)$ at times $(t, t+1)$, the joint likelihood factors as

$$p(x, z) = \prod_{r=1}^R \left[ p(z_{r,1}) \prod_{t} \prod_{n \in C_{r,t}} p\bigl(x_{n,t} \mid z_{n,t}, x_{n,t-1}\bigr) \prod_{c_n} p\bigl(z_{c_n, t+1} \mid z_{n,t}\bigr) \right]$$

where the transition term $p(z_{c_n, t+1} \mid z_{n,t})$ is given by:
- the standard transition matrix $P$ for **non-division** (self) transitions;
- the division transition matrix $\tilde P$ for **division** events.

See the **Derivation** folder for forward–backward details. 

---

## TODOs

- Optionally, division events themselves can be modeled by appending a division indicator to the observations.  
- Tests  
- Improve sampling and demonstration notebook  

---

See the notebook for usage.
