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

---

## Closed Form by Group

### By technology

Nuclear (ranks 0–1) and gas (ranks 2–3) are each two **consecutive** generators, so their
sum collapses to a single clamp:

$$
\text{nuclear}(t,s) = g_0 + g_1 = \min\bigl(L(t,s),\ 500\bigr)
$$

$$
\text{gas}(t,s) = g_2 + g_3 = \max\bigl(L(t,s) - 500,\ 0\bigr)
$$

### By company

Britishnuke owns only $g_1$, so:

$$
\text{britishnuke}(t,s) = \max\!\bigl(\min\bigl(L(t,s) - 250,\ 250\bigr),\ 0\bigr)
$$

Rhonepower owns $g_0$, $g_2$, $g_3$ — **non-consecutive** ranks, so no further collapse:

$$
\text{rhonepower}(t,s) = \min\bigl(L(t,s),\ 250\bigr) + \max\bigl(L(t,s) - 500,\ 0\bigr)
$$

**Sanity checks** — both partitions sum to $L(t,s)$:

$$
\text{nuclear} + \text{gas} = \min(L, 500) + \max(L-500, 0) = L \quad \checkmark
$$

$$
\text{rhonepower} + \text{britishnuke} = g_0 + g_1 + g_2 + g_3 = L \quad \checkmark
$$
