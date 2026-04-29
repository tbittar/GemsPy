# Basic Example — Closed-Form Generator Productions

## Setup

| Generator | Rank $k$ | Cost (€/MWh) | $P_\max$ (MW) | Technology | Company      |
|-----------|----------|--------------|----------------|------------|--------------|
| nuclear_1 | 0        | 5            | 250            | nuclear    | rhonepower   |
| nuclear_2 | 1        | 10           | 250            | nuclear    | britishnuke  |
| gas_1     | 2        | 50           | 250            | gas        | rhonepower   |
| gas_2     | 3        | 100          | 250            | gas        | rhonepower   |

Load $L(t, s) \in [0, 1000]$ MW.  Total capacity $= 4 \times 250 = 1000$ MW.

---

## Closed Form

Since costs are strictly ordered and there are no commitment constraints, the LP reduces to
pure merit-order dispatch. Generator at rank $k$ produces:

$$
\boxed{g_k(t,s) = \operatorname{clamp}\!\bigl(L(t,s) - k\,P_\max,\ 0,\ P_\max\bigr)}
$$

Explicitly for each generator ($P_\max = 250$ MW):

$$
g_0(t,s) = \min\bigl(L(t,s),\ 250\bigr)
$$

$$
g_1(t,s) = \max\!\bigl(\min\bigl(L(t,s) - 250,\ 250\bigr),\ 0\bigr)
$$

$$
g_2(t,s) = \max\!\bigl(\min\bigl(L(t,s) - 500,\ 250\bigr),\ 0\bigr)
$$

$$
g_3(t,s) = \max\!\bigl(\min\bigl(L(t,s) - 750,\ 250\bigr),\ 0\bigr)
$$

No unsupplied energy or spillage since $L(t,s) \in [0, N P_\max] = [0, 1000]$.
