# linopy (and xarray) API surface used by GemsPy

This document enumerates every linopy entry point (class, method,
attribute, operator overload) used by the GemsPy codebase, together
with the xarray surface on which those entry points rely. linopy is
GemsPy's symbolic LP/MILP layer; xarray is the underlying data layout.
The two cannot really be separated when documenting the integration —
linopy variables are xarray-shaped, bounds are `xr.DataArray`s, and
the post-solve `Model.solution` is an `xr.Dataset` — so both are
catalogued here.

Unless stated otherwise, file paths are relative to the repository
root and line numbers refer to the state of the branch on which this
document was produced.

---

## 1. Overview

GemsPy builds **one** `linopy.Model` per `OptimizationProblem`
(`src/gems/simulation/optimization.py`) and exposes a high-level API
(`solve`, `status`, `objective_value`, `export_lp`,
`get_variable_labels`) that wraps that single model.

linopy and xarray are intertwined in GemsPy in three ways:

1. **Inputs to linopy are xarray objects.** Parameter arrays, bounds
   and `coords` passed to `Model.add_variables` are `xr.DataArray`s.
2. **linopy outputs are xarray objects.** `Model.solution` is an
   `xr.Dataset`, `Variable.labels` is an `xr.DataArray`, and arithmetic
   on Variables produces `LinearExpression` objects that share the
   xarray dim/coord machinery.
3. **Arithmetic mixes both worlds.** Constraint LHS expressions are
   built by multiplying linopy Variables by `xr.DataArray` parameters
   and summing along xarray dimensions; the operand-order quirk
   between `xr.DataArray.__add__` and `linopy.LinearExpression.__add__`
   is the reason GemsPy ships the `_linopy_add` helper.

### Files that directly import `linopy`

| File | Role |
|---|---|
| `src/gems/simulation/optimization.py` | Builds the linopy model, runs the solver, exposes results. |
| `src/gems/simulation/linearize.py` | Walks the GemsPy AST and produces `linopy.LinearExpression`s. |
| `src/gems/simulation/vectorized_builder.py` | Vectorized expression builder; defines `VectorizedExpr` and `_linopy_add`. |
| `src/gems/simulation/simulation_table.py` | Reads `Model.solution` and turns it into a simulation table. |
| `src/gems/session/session.py` | Reads `Model.solution` to compute carry-over values. |
| `tests/unittests/simulation/test_vectorized_linear_expr_builder.py` | Unit tests around `_linopy_add` and the vectorized builder. |
| `tests/e2e/functional/test_out_of_bounds_processing.py` | E2E test that inspects `Model.constraints[...]`. |

### Files that depend on xarray but not on linopy directly

These still belong to the same data plane (they manipulate xarrays
that come out of, or feed into, the linopy layer):

- `src/gems/simulation/extra_output.py`
- `src/gems/simulation/couplings.py`

---

## 2. linopy top-level symbols

### 2.1 `linopy.Model`

- Construction: `linopy.Model()`
    - `optimization.py:403` (problem builder)
    - `tests/unittests/simulation/test_vectorized_linear_expr_builder.py:53`
- Alias: `LinopyModel = linopy.Model` (`optimization.py:63`)
- xarray side: a `Model` keeps its variables/constraints indexed by
  xarray dims. The only xarray-typed public surface of `Model` that
  GemsPy touches is `Model.solution` (an `xr.Dataset`).

### 2.2 `linopy.Variable`

- Used as a type annotation and as the runtime object cached in
  `OptimizationProblem._linopy_vars: Dict[(str, str), linopy.Variable]`.
    - `optimization.py:308`, `404`, `433`
    - `linearize.py:77`, `83`
    - `vectorized_builder.py:77`
- xarray side: a `Variable` behaves like an `xr.DataArray` of integer
  labels and exposes `.name`, `.dims`, `.coords`, `.sizes`,
  `.labels`, `.isel`, `.sel`, `.sum`, `.where` (see §4).

### 2.3 `linopy.LinearExpression`

- Returned by arithmetic on linopy Variables (and by mixing Variables
  with `xr.DataArray`s).
    - `vectorized_builder.py:77` — in the `VectorizedExpr` union with
      `xr.DataArray` and `linopy.Variable`.
    - `tests/unittests/simulation/test_vectorized_linear_expr_builder.py`
      — `isinstance(..., linopy.LinearExpression)` checks at lines
      `123`, `132`, `142`, `150`, `262`.
- xarray side: same `.dims` / `.coords` semantics as `Variable`;
  consumed via `.sum`, `.isel`, and mixed arithmetic with DataArrays.

---

## 3. `linopy.Model` — methods and attributes used

| Entry point | Call sites | xarray dependency |
|---|---|---|
| `Model()` | `optimization.py:403`; `test_vectorized_linear_expr_builder.py:53` | — |
| `Model.add_variables(lower, upper, coords, name, binary, integer)` | `optimization.py:455` (dummy: `lower=xr.DataArray([0.0], dims=["__dummy_dim"])`, `upper=xr.DataArray([0.0], dims=["__dummy_dim"])`); `optimization.py:615` (real variables: numeric `lower`/`upper`, `coords` dict `dim → list/Index`, optional `binary`/`integer`); `test_vectorized_linear_expr_builder.py:63` | `lower`/`upper` accepted as `xr.DataArray`; `coords` is the xarray coords mapping that the returned `Variable` carries forever. |
| `Model.add_constraints(<comparison>, name=...)` | `optimization.py:436` (carry-over equality: `linopy_var.isel(time=0) == init_val`); `optimization.py:701` (lower bound: `lhs >= lb`); `optimization.py:710` (upper bound: `lhs <= ub`) | LHS is a `LinearExpression` with xarray dims; RHS is an `xr.DataArray`; the comparison broadcasts along shared dims. |
| `Model.add_objective(<expr>)` | `optimization.py:446` (real objective); `optimization.py:460` (`0 * dummy` so a model with no real variables still solves) | input is a `LinearExpression` carrying xarray dims (or a scaled dummy `Variable`). |
| `Model.solve(solver_name, **kwargs)` | `optimization.py:331` (default `solver_name="highs"`, with `io_api="direct"` defaulted in via `kwargs.setdefault`) | populates `Model.solution` (`xr.Dataset`). |
| `Model.to_file(path, explicit_coordinate_names=True)` | `optimization.py:350` | uses xarray coord names for a human-readable LP file. |
| `Model.status` | `optimization.py:336` | — |
| `Model.termination_condition` | `optimization.py:341` | — |
| `Model.objective.value` | `optimization.py:346` | — |
| `Model.variables` | `optimization.py:454` (`len(...)` to know whether to add a dummy variable); `optimization.py:636` (cross-check against GemsPy-side dict) | — |
| `Model.constraints[name]` | `tests/e2e/functional/test_out_of_bounds_processing.py:83` (`con.lhs.vars.values` — numpy-like access for assertions) | the returned `Constraint`'s `.lhs.vars.values` is a numpy/xarray array. |
| `Model.solution` | `session.py:216`; `simulation_table.py:231`, `270` | returns an `xr.Dataset`; then accessed as `solution[lv.name] -> xr.DataArray` (`session.py:221`, `simulation_table.py:239`, `274`). |

---

## 4. `linopy.Variable` / `linopy.LinearExpression` — attributes and methods

These are essentially the xarray API exposed on linopy objects. Each
row lists what is used in GemsPy and what it gives back.

| Member | Returns | Call sites |
|---|---|---|
| `.name` | `str` | `optimization.py:544`, `611`, `623`; `simulation_table.py:236`, `239`, `273`, `274`; `session.py:220`, `221` |
| `.dims` | tuple of dim names | `optimization.py:127`, `434`; `vectorized_builder.py:219`, `224`, `450`, `483`, `488`; `session.py:220`; `simulation_table.py:304`, `365`, `366`, `368` |
| `.coords` | xarray `Coordinates` mapping | `simulation_table.py:240` (`lv.coords["component"].values`), `307`; `vectorized_builder.py:402` |
| `.sizes` | `Mapping[dim, int]` | `vectorized_builder.py:323` (`operand.sizes["scenario"]`) |
| `.labels` | `xr.DataArray` of integer labels | `optimization.py:363` (returned to callers by `get_variable_labels`); consumed by `simulation/couplings.py:49`, `65` via `.sel(component=…).item()`. |
| `.isel(time=…)` | sliced expression | `optimization.py:132`, `437`; `vectorized_builder.py:240`, `401` |
| `.sel(component=…)` | label-selected expression / value | `simulation_table.py:241`, `309`; `couplings.py:49`, `65` |
| `.sum(dim)` | reduction along `dim` | `optimization.py:289` (`.sum("component_master")`), `725`, `727`; `vectorized_builder.py:308` (`.sum("time")`), `323` (`.sum("scenario")`) |
| `.where(mask)` | mask-restricted expression | `optimization.py:129` (used inside `_apply_validity_mask` when the mask is broadcastable) |
| `.expand_dims`, `.transpose`, `.assign_coords`, `.rename` | dim manipulation | `optimization.py:288` (`.rename({"component": "component_master"})`); `simulation_table.py:369`, `371`, `373`, `375`; `vectorized_builder.py:403` (`.assign_coords(time=…)`) |
| `.values`, `.astype(...)` | numpy/array conversion | `optimization.py:131`, `760`; `vectorized_builder.py:225`, `226`, `282`, `283`, `422`; `simulation_table.py:307`, `376`, `398` |
| Comparison operators (`==`, `<=`, `>=`) | constraint expression | `optimization.py:437`, `700`, `709` |
| Arithmetic operators (`+`, `-`, `*`) | linear expression | `optimization.py:289` (`A * expr_master_r`), `291`; `vectorized_builder.py:232`, `258`, `302`; `linearize.py:106` — see `_linopy_add` (§5) for the operand-swap quirk. |

---

## 5. xarray API surface used in support of linopy

These calls are not on linopy objects, but they either produce
inputs that flow into linopy or post-process linopy results, so they
belong in the linopy/xarray integration picture.

| xarray entry point | Call sites | Purpose |
|---|---|---|
| `xr.DataArray(...)` constructor | `optimization.py:207`, `257`, `271`, `293`, `456`, `457`, `530`; `vectorized_builder.py:153`, `263`, `268`, `293`, `298`, `354`, `400` | Build `lower`/`upper`, parameter arrays, port arrays, and time indexers fed into linopy. |
| `xr.Dataset` (type only, plus `Dataset.from_dataframe`) | `simulation_table.py:141`, `159` | Type of `Model.solution`; also used to materialize simulation results. |
| `xr.where(cond, a, b)` | `linearize.py:136`, `148`; `vectorized_builder.py:375`, `382`, `531`, `537` | Element-wise piecewise / min / max combinations on the scalar/parameter branch, before linopy sees the result. |
| `xr.align(a, b, join="exact")` | `extra_output.py:73` | Reconcile two post-solve `xr.DataArray`s before equality comparison. |
| Mixed `DataArray`/linopy arithmetic (`+`, `*`, `==`, `<=`, `>=`) | `optimization.py:289`, `437`, `700`, `709`; `vectorized_builder.py:92–105`; `linearize.py:97–106` | `xr.DataArray.__add__` does not recognise linopy types, so `DataArray + LinearExpression` would raise. `_linopy_add` and the AST builder's `add` override always put the linopy operand on the left so that `linopy.__add__` / `__radd__` is used. |
| `Model.solution[name]` then `.isel(time=…, drop=True)` | `session.py:221`, `222` | Extract carry-over values from the solution. |

---

## 6. End-to-end data flow

The list above can feel disjointed; the integrated story is:

1. **Parameters as DataArrays.** GemsPy converts study data to
   `xr.DataArray`s indexed by `(component, time, scenario)` — see
   `_build_param_arrays` in `optimization.py` (≈ lines `485–540`).

2. **Decision variables as linopy Variables.** For each model
   variable, `Model.add_variables(lower=..., upper=..., coords=...,
   name=..., binary=..., integer=...)` is called
   (`optimization.py:615`). The returned `Variable` is itself
   xarray-shaped.

3. **Building expressions.** `VectorizedLinearExprBuilder`
   (`linearize.py`) walks the GemsPy AST, resolving variable nodes
   via `linopy_vars[(model_id, var_name)]` and multiplying them by
   `xr.DataArray` parameters. The accumulator uses `_linopy_add`
   (`vectorized_builder.py:92`) to deal with operand ordering.

4. **Constraints.** For each model constraint, GemsPy builds
   `lhs <= ub` and `lhs >= lb` where `lhs` is a `LinearExpression`
   and `ub`/`lb` are `xr.DataArray`s, then registers the result via
   `Model.add_constraints(..., name=...)` (`optimization.py:700–710`).
   Carry-over constraints between blocks use the `==` form
   (`optimization.py:437`).

5. **Objective.** Per-model contributions are accumulated with
   `_linopy_add`; any pure-DataArray contribution is collapsed to a
   scalar offset (`_objective_constant`) and the linopy-typed total
   is registered with `Model.add_objective(...)`
   (`optimization.py:441–446`).

6. **Solve.** `OptimizationProblem.solve` forwards to
   `Model.solve(solver_name=..., io_api="direct", **kwargs)`
   (`optimization.py:325–331`).

7. **Reading results.** Status, termination and objective value are
   exposed by simple property accesses
   (`optimization.py:333–346`). Variable values come back through
   `problem.linopy_model.solution[var_name]` (`session.py:216`,
   `simulation_table.py:231`, `270`); the resulting `xr.DataArray`s
   are sliced (`.isel`, `.sel`), reshaped (`.expand_dims`,
   `.transpose`), and emitted as a tabular simulation result
   (`simulation_table.py:352+`).

---

## 7. Ouverture — entry points GemsPy could reasonably use next

Some linopy and xarray features are not used today but would be
natural extensions of the integration. Listed roughly in order of
expected payoff.

### linopy

- **`Model.add_variables(mask=…)`** — partial-domain variables.
  Would replace the current `_apply_validity_mask`
  (`optimization.py:118–132`) trick, which currently rebuilds an
  expression via `.isel(time=valid_times)` or `.where(mask)`.
- **`Variable.shift(time=…)` / `.roll(...)`** — built-in time shifting.
  Would replace the hand-written `_apply_time_shift`
  (`vectorized_builder.py:392+`) which manually builds an `xr.DataArray`
  of positions and does `operand.isel(time=indexer)`.
- **`Model.add_constraints(lhs, sign, rhs, name=…)` (separate-arguments
  form)** — a cleaner pattern for the double-sided bound case that
  GemsPy currently expresses as two operator-based `add_constraints`
  calls (`optimization.py:701` and `:710`).
- **`Model.remove_constraints` / `remove_variables`** — enable
  incremental rebuilds and warm restarts for rolling-horizon studies,
  instead of always rebuilding from scratch.
- **`Model.add_objective(..., overwrite=True)`** — replace, rather than
  accumulate, an objective when re-optimizing the same model.
- **`Model.compute_infeasibilities()` /
  `print_infeasibilities()`** — IIS diagnostics for infeasible runs;
  would slot naturally into the
  `tests/e2e/functional/test_out_of_bounds_processing.py` pipeline.
- **`Model.to_netcdf()` / `Model.from_netcdf()`** — persist a fully
  built model. Today only LP-format export through `Model.to_file`
  (`optimization.py:350`) is exposed.
- **`Variable.solution` / `Variable.dual`,
  `LinearExpression.solution`** — per-variable solution and dual
  accessors. GemsPy currently always goes through
  `Model.solution[name]`; per-variable access would simplify the
  carry-over and simulation-table code paths.
- **`linopy.merge(...)`** — combine expressions across subproblems;
  a natural fit for Benders-style decomposition on top of the
  existing master/sub setup (`optimization.py:271–293`).
- **Solver-specific kwargs forwarded through `Model.solve(**kwargs)`**
  — HiGHS presolve, Gurobi MIP gap, Mosek params, etc. The plumbing
  already exists (`optimization.py:325–331`); only documentation and
  defaults would be needed.

### xarray (linopy-adjacent)

- **`DataArray.broadcast_like` / `xr.broadcast`** — replace the
  manual `np.broadcast_to` / `np.expand_dims` dance in
  `optimization.py:757–764`.
- **`DataArray.stack` / `unstack`** — multi-index handling when
  assembling the simulation table; today this is done via pandas
  pivots in `simulation_table.py:141–164`.
- **`xr.concat`** — combine per-block solutions directly, instead
  of going through intermediate Python dicts.
- **`DataArray.interp`** — interpolate coarser exogenous time series
  onto the optimization horizon.
- **`xr.merge`** — combine `Model.solution` with the extra-output
  arrays built by `VectorizedExtraOutputBuilder`
  (`extra_output.py:83–111`) into a single output dataset.

---

## 8. Cross-reference by file

Quick per-file index so a reader can jump in.

### `src/gems/simulation/optimization.py`
- `linopy.Model()`, `Model.add_variables`, `Model.add_constraints`,
  `Model.add_objective`, `Model.solve`, `Model.to_file`,
  `Model.status`, `Model.termination_condition`,
  `Model.objective.value`, `Model.variables`.
- `Variable.dims`, `.name`, `.labels`, `.isel`.
- `xr.DataArray` constructor (bounds, parameter arrays, port
  contributions); `DataArray.sum`, `.rename`, `.where`, `.values`,
  `.dims`, `.coords`.

### `src/gems/simulation/linearize.py`
- `linopy.Variable` type annotation; resolution from
  `linopy_vars` dict.
- Operand-swap addition with `_linopy_add` for mixing DataArray and
  linopy objects.
- `xr.where` for piecewise / min / max on the scalar branch.

### `src/gems/simulation/vectorized_builder.py`
- `linopy.Variable`, `linopy.LinearExpression` (in the
  `VectorizedExpr` union).
- `_linopy_add` helper — the canonical place where the
  `DataArray + linopy` operand-ordering issue is handled.
- `.isel`, `.sum`, `.sizes`, `.dims`, `.assign_coords`, `.coords`
  on whichever member of `VectorizedExpr` is current.
- `xr.DataArray` constructors and `xr.where`.

### `src/gems/simulation/simulation_table.py`
- `Model.solution` (xr.Dataset) → `solution[lv.name]`
  (xr.DataArray).
- `Variable.name`, `Variable.coords["component"].values`.
- `DataArray.sel`, `.expand_dims`, `.transpose`, `.values`, `.dims`.
- `xr.Dataset.from_dataframe`.

### `src/gems/session/session.py`
- `problem.linopy_model.solution`.
- `Variable.dims`, `Variable.name`.
- `DataArray.isel(time=local_index, drop=True)` to extract the
  carry-over slice.

### `src/gems/simulation/extra_output.py`
- xarray only: `xr.align(..., join="exact")`, `DataArray.values`.
  No direct `linopy` import, but it processes arrays that came out
  of `Model.solution`.

### `src/gems/simulation/couplings.py`
- xarray only: consumes the `xr.DataArray` returned by
  `OptimizationProblem.get_variable_labels` (i.e. `Variable.labels`)
  via `.sel(component=...).item()` (`couplings.py:49`, `65`).

### `tests/unittests/simulation/test_vectorized_linear_expr_builder.py`
- `linopy.Model()`, `Model.add_variables(lower, upper, coords,
  name)`.
- `isinstance` checks on `linopy.LinearExpression` and
  `linopy.Variable`.

### `tests/e2e/functional/test_out_of_bounds_processing.py`
- `linopy.Model` type hint.
- `Model.constraints[name]` → `Constraint.lhs.vars.values` for
  assertion-time inspection.
