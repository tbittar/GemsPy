# Basic Example — Closed-Form Expected Productions

## Setup

| Generator | Rank $k$ | Cost (€/MWh) | $P_\max$ (MW) | Technology | Company      |
|-----------|----------|--------------|----------------|------------|--------------|
| nuclear_1 | 0        | 5            | 250            | nuclear    | rhonepower   |
| nuclear_2 | 1        | 10           | 250            | nuclear    | britishnuke  |
| gas_1     | 2        | 50           | 250            | gas        | rhonepower   |
| gas_2     | 3        | 100          | 250            | gas        | rhonepower   |

Load $L(t, s)$: drawn uniformly on $[0, 1000]$ MW (30 timesteps × 10 scenarios, full range covered).

Total capacity: $N \cdot P_\max = 4 \times 250 = 1000$ MW.

---

## Merit-Order Dispatch

Since costs are strictly increasing and there are no commitment constraints, the LP solution is
pure merit-order at every $(t, s)$:

$$
g_k(t,s) = \operatorname{clamp}\!\bigl(L(t,s) - k\,P_\max,\; 0,\; P_\max\bigr)
$$

where $\operatorname{clamp}(x, a, b) = \max(a, \min(x, b))$.

No unsupplied energy or spillage occurs because $L \in [0, 1000] = [0, N P_\max]$.

---

## Closed Form for Each Generator

With $L \sim \mathcal{U}(0,\, N P_\max)$, generator at rank $k$ exclusively covers the load
segment $[k P_\max,\, (k+1) P_\max]$, giving:

$$
\boxed{
\mathbb{E}[g_k] = P_\max \cdot \frac{2(N-k)-1}{2N}
}
$$

This is an arithmetic sequence in $k$, with step $-P_\max / N$:

| Generator | $k$ | Formula                          | Value (MW) |
|-----------|-----|----------------------------------|------------|
| nuclear_1 | 0   | $250 \times \tfrac{7}{8}$        | **218.75** |
| nuclear_2 | 1   | $250 \times \tfrac{5}{8}$        | **156.25** |
| gas_1     | 2   | $250 \times \tfrac{3}{8}$        |  **93.75** |
| gas_2     | 3   | $250 \times \tfrac{1}{8}$        |  **31.25** |

**Total** = 500 MW = $\mathbb{E}[L]$ ✓

---

## Expected Production by Technology

$$
\mathbb{E}[\text{nuclear}]
  = \mathbb{E}[g_0] + \mathbb{E}[g_1]
  = P_\max\!\cdot\frac{7+5}{8}
  = 250 \times \frac{3}{4} = \mathbf{375 \text{ MW}}
$$

$$
\mathbb{E}[\text{gas}]
  = \mathbb{E}[g_2] + \mathbb{E}[g_3]
  = P_\max\!\cdot\frac{3+1}{8}
  = 250 \times \frac{1}{4} = \mathbf{125 \text{ MW}}
$$

Nuclear covers the lower $[0, 500]$ MW band and gas the upper $[500, 1000]$ MW band, so
nuclear always runs more: $\mathbb{E}[\text{nuclear}] = 3\,\mathbb{E}[\text{gas}]$.

---

## Expected Production by Company

$$
\mathbb{E}[\text{rhonepower}]
  = \mathbb{E}[g_0] + \mathbb{E}[g_2] + \mathbb{E}[g_3]
  = P_\max\!\cdot\frac{7+3+1}{8}
  = 250 \times \frac{11}{8} = \mathbf{343.75 \text{ MW}}
$$

$$
\mathbb{E}[\text{britishnuke}]
  = \mathbb{E}[g_1]
  = P_\max\!\cdot\frac{5}{8}
  = \mathbf{156.25 \text{ MW}}
$$
