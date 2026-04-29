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
pure merit-order dispatch. Generator at rank `k` produces:

```python
def g(k, L, p_max=250):
    return max(min(L - k * p_max, p_max), 0)
```

Explicitly for each generator:

```python
nuclear_1 = min(L, 250)
nuclear_2 = max(min(L - 250, 250), 0)
gas_1     = max(min(L - 500, 250), 0)
gas_2     = max(min(L - 750, 250), 0)
```

No unsupplied energy or spillage since `L` ∈ `[0, 1000]`.

---

## Closed Form by Group

### By technology

Nuclear (ranks 0–1) and gas (ranks 2–3) are each two **consecutive** generators, so their
sum collapses to a single clamp:

```python
nuclear = min(L, 500)
gas     = max(L - 500, 0)
```

### By company

Britishnuke owns only `nuclear_2`:

```python
britishnuke = max(min(L - 250, 250), 0)
```

Rhonepower owns `nuclear_1`, `gas_1`, `gas_2` — **non-consecutive** ranks, no further collapse:

```python
rhonepower = min(L, 250) + max(L - 500, 0)
```

**Sanity checks** — both partitions sum to `L`:

```python
assert nuclear + gas == L                    # min(L,500) + max(L-500,0) == L  ✓
assert rhonepower + britishnuke == L         # g0 + g1 + g2 + g3 == L          ✓
```
