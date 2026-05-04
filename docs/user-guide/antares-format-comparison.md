# GemsPy vs Antares Simulator: Study Format Comparison

This document compares the study format expected by GemsPy with the full GEMS format
implemented in [Antares Simulator v10.1.0](https://github.com/AntaresSimulatorTeam/Antares_Simulator/releases/tag/v10.1.0),
as documented in the
[Antares Simulator modeler inputs reference](https://github.com/AntaresSimulatorTeam/Antares_Simulator/blob/develop/docs/user-guide/modeler/02-inputs.md).

Only features that differ between the two tools are listed.

---

## 1. Study Directory Structure

| Path | GemsPy | Antares Simulator v10.1.0 |
|---|---|---|
| Simulation parameters file | `input/optim-config.yml` | `parameters.yml` *(at study root)* |
| Scenario builder file | `input/scenariobuilder.dat` | `input/data-series/modeler-scenariobuilder.dat` |

**GemsPy layout:**
```
study/
├── input/
│   ├── optim-config.yml           ← simulation parameters
│   ├── system.yml
│   ├── scenariobuilder.dat        ← optional
│   ├── model-libraries/
│   └── data-series/
└── output/
```

**Antares layout:**
```
study/
├── parameters.yml                 ← simulation parameters (at root)
├── input/
│   ├── optim-config.yml           ← optional, hybrid studies only
│   ├── system.yml
│   ├── model-libraries/
│   └── data-series/
│       └── modeler-scenariobuilder.dat  ← optional, inside data-series/
└── output/
```

---

## 2. Simulation Parameters

GemsPy uses `input/optim-config.yml`; Antares uses `parameters.yml` at the study root.

| Feature | GemsPy (`input/optim-config.yml`) | Antares (`parameters.yml`) |
|---|---|---|
| Solver name field | `solver-options.name` (default: `highs`) | `solver` (**required**, no default) |
| Supported solvers | `highs` only | `sirius`, `scip`, `coin`, `xpress`, `glpk`, `highs`, `pdlp`, `gurobi` |
| Enable solver logs | `solver-options.logs` | `solver-logs` |
| Solver extra parameters | `solver-options.parameters` (all solvers) | `solver-parameters` (SCIP/XPRESS/PDLP only) |
| First timestep | `time-scope.first-time-step` | `first-time-step` |
| Last timestep | `time-scope.last-time-step` (default: `0`) | `last-time-step` (default: `167`) |
| Number of scenarios | `scenario-scope.nb-scenarios` (default: `1`) | Not present (derived from data) |
| Suppress output files | Not present | `no-output` (default: `false`) |
| Export MPS file | Not present | `export-mps` (default: `false`) |
| Resolution mode | `resolution.mode` (see §3) | In hybrid `optim-config.yml` only |
| Rolling window length | `resolution.block-length` | Not present |
| Rolling window overlap | `resolution.block-overlap` (default: `0`) | Not present |
| Per-model decomposition | `models[].model-decomposition` (all studies) | `models[].model-decomposition` (hybrid `optim-config.yml` only) |
| Out-of-bounds handling | `models[].out-of-bounds-processing` (all studies) | `models[].out-of-bounds-processing` (hybrid `optim-config.yml` only) |

---

## 3. Resolution Modes

| Mode | GemsPy | Antares |
|---|---|---|
| `frontal` | Yes *(default)* — single optimisation problem over the full horizon | No |
| `sequential-subproblems` | Yes | Yes *(Antares default for hybrid studies)* |
| `parallel-subproblems` | Yes — rolling window, blocks solved in parallel | No |
| `benders-decomposition` | Yes | Yes *(hybrid studies only)* |

`block-length` and `block-overlap` are GemsPy-only parameters that control the rolling
window for `sequential-subproblems` and `parallel-subproblems` modes.

---

## 4. Model Library Format

### 4a. Library-level fields

| Field | GemsPy | Antares |
|---|---|---|
| `library.models` | Optional (defaults to `[]`) | **Required** (no fallback in parser; absent key throws) |
| `library.dependencies` | Supported — list of library IDs | **Not present** — no field in struct or parser |

### 4b. Parameter defaults

Source: GemsPy `ParameterSchema` field defaults; Antares `decoders.cpp` `as<bool>(default)` calls.

| Field | GemsPy default | Antares default |
|---|---|---|
| `parameters[].time-dependent` | `false` | `true` |
| `parameters[].scenario-dependent` | `false` | `true` |

### 4c. Variable type names

| Variable type | GemsPy | Antares |
|---|---|---|
| Boolean/Binary | `binary` | `boolean` |

### 4d. Constraint syntax

GemsPy supports two syntaxes; Antares supports only the inline comparison form.

| Aspect | GemsPy | Antares |
|---|---|---|
| Separate `lower-bound` / `upper-bound` fields | Supported (GemsPy extension) | Not present |
| Mixing inline comparison with separate bounds | Error | N/A |

GemsPy — inline comparison (compatible with Antares):
```yaml
constraints:
  - id: flow_limit
    expression: flow <= capacity
```

GemsPy — separate bounds (GemsPy-only extension):
```yaml
constraints:
  - id: flow_limit
    expression: flow
    lower-bound: -capacity
    upper-bound: capacity
```

---

## 5. System File (`system.yml`)

| Feature | GemsPy | Antares |
|---|---|---|
| `system.model-libraries` field | Parsed but **unused** — libraries auto-loaded from `input/model-libraries/` | Parsed and used to enumerate libraries |
| `system.area-connections` | Parsed but **silently dropped** — never passed to any resolver | Parsed and supported (hybrid studies) |

---

## 6. Data Series Files

| Feature | GemsPy | Antares |
|---|---|---|
| Supported file extensions | `.txt`, `.tsv` | `.csv`, `.tsv` |
| Column separator for `.tsv` | Tab only | Tab or space |

---

## 7. Scenario Builder

| Feature | GemsPy | Antares |
|---|---|---|
| File name | `scenariobuilder.dat` | `modeler-scenariobuilder.dat` |
| File location | `input/` | `input/data-series/` |

---

## 8. Expression Language

### 8a. Grammar: power operator

| Feature | GemsPy | Antares v10.1.0 |
|---|---|---|
| Infix exponentiation `x ^ y` | Not supported — parse error | Supported (right-associative, higher precedence than `*`/`/`) |

Antares adds `^` to three grammar rules (`expr`, `shift_expr`, `right_expr`) so it also works inside time-shift contexts (e.g. `x[t + 2^n]`). GemsPy's grammar has no `^` token.

```yaml
# Antares v10.1.0 — valid
expression: efficiency ^ 2 * capacity

# GemsPy — ANTLR parse error: ^ is not in the grammar
expression: efficiency ^ 2 * capacity
```

In linear contexts (constraints, objectives, variable bounds) both operands of `^` must reduce to literals or parameters — not decision variables. Variables are only permitted inside `extra-output` expressions.

### 8b. Named functions

| Function | GemsPy | Antares v10.1.0 |
|---|---|---|
| `expec(x)` | Supported (see §9) | Not implemented in v10.1.0 — parse error (see §9) |
| `reduced_cost(var)` | Not supported | Supported — `extra-output` and `port-field-definitions` only |
| `dual(constraint)` | Not supported | Supported — `extra-output` and `port-field-definitions` only |
| `max(a, b, ...)` | Supported; no parse-time restriction on argument type | Supported; arguments must not contain decision variables in linear contexts |
| `min(a, b, ...)` | Supported; no parse-time restriction on argument type | Supported; arguments must not contain decision variables in linear contexts |
| `floor(x)` | Supported; no parse-time restriction on argument type | Supported; argument must not contain decision variables in linear contexts |
| `ceil(x)` | Supported; no parse-time restriction on argument type | Supported; argument must not contain decision variables in linear contexts |

`reduced_cost(var)` and `dual(constraint)` expose LP post-optimisation data (reduced costs and shadow prices). Antares makes them available only in `extra-output` expressions (and transitively through port field connections). GemsPy has no equivalent functions.

Antares enforces the variable-exclusion rule for `max`/`min`/`floor`/`ceil` at parse/validation time via a `ForbiddenNodesVisitor` that checks each expression against a per-context rule set. GemsPy has no equivalent parse-time check — a non-linear combination would only fail later when the LP is assembled.

---

## 9. `expec()` Operator

`expec(X)` computes the expected value (scenario-wise average) of a scenario-dependent
expression.

| Aspect | GemsPy | Antares v10.1.0 |
|---|---|---|
| `expec()` in **objective-contributions** | **Supported** (explicit) | **Forbidden** |
| Scenario-dependent objective without `expec()` | Auto-wrapped with `expec()` + `UserWarning` (compat shim) | Averaged implicitly by the solver |

In Antares v10.1.0, scenario-dependent objective contributions are averaged over scenarios
internally — authors do not write `expec()` explicitly, and doing so is a syntax error.

In GemsPy, `expec()` must eventually be written explicitly. As a temporary compatibility
shim for studies originally written for Antares v10.0.0, GemsPy auto-wraps any
scenario-dependent objective contribution that lacks `expec()` and emits a `UserWarning`.
Once Antares natively supports `expec()` in objectives this shim will be removed.

```yaml
# Antares v10.1.0 — expec() NOT allowed in objectives
objective-contributions:
  - id: expected_cost
    expression: cost * output   # scenario-dependent; averaged implicitly

# GemsPy — expec() REQUIRED (or warned and auto-applied)
objective-contributions:
  - id: expected_cost
    expression: expec(cost * output)  # explicit; no warning
```

---

## 10. `model-decomposition` in `optim-config`

Both tools support a `model-decomposition` block inside `optim-config.yml` that assigns
variables, constraints, and objective contributions to a location in the Benders
decomposition (`master`, `subproblems`, or `master-and-subproblems`).

| Aspect | GemsPy | Antares v10.1.0 |
|---|---|---|
| `models` list in `optim-config.yml` mandatory | No (`models: []` by default) | **Yes** — required when the file is present (no fallback in parser) |
| Antares `optim-config.yml` content | Resolution/solver/time/scenario config unified in one file | Only `resolution-mode` + `models`; solver/time/scenario are in `parameters.yml` |
| Applicable resolution modes | `benders-decomposition` (meaningful); accepted but ignored in other modes | `benders-decomposition` and `sequential-subproblems` only |
| Default resolution mode | `frontal` | `sequential-subproblems` |

The `model-decomposition` structure is identical in both tools:

```yaml
# optim-config.yml (both GemsPy and Antares)
models:
  - id: my_lib.storage
    model-decomposition:
      variables:
        - id: stored_energy
          location: master
      constraints:
        - id: storage_balance
          location: master
      objective-contributions:
        - id: investment_cost
          location: master
```

---

## 11. Feature Summary Table

| Feature | GemsPy | Antares v10.1.0 |
|---|---|---|
| `frontal` resolution (single-horizon LP) | Yes *(default)* | No |
| `parallel-subproblems` resolution (rolling window, parallel) | Yes | No |
| Rolling window `block-overlap` | Yes | No |
| Explicit `nb-scenarios` in config | Yes | No |
| `no-output` flag | No | Yes |
| `export-mps` flag | No | Yes |
| Library `dependencies` field | Yes | No |
| Library `models` field mandatory | No | Yes |
| Boolean variable type keyword | `binary` | `boolean` |
| Parameter `time-dependent` default | `false` | `true` |
| Parameter `scenario-dependent` default | `false` | `true` |
| Constraint separate `lower-bound`/`upper-bound` fields | Yes | No |
| Data series `.txt` extension | Yes | No |
| Data series `.csv` extension | No | Yes |
| `system.model-libraries` field used | No (silently ignored) | Yes |
| `system.area-connections` implemented | No (silently dropped) | Yes |
| Scenario builder filename | `scenariobuilder.dat` | `modeler-scenariobuilder.dat` |
| Scenario builder location | `input/` | `input/data-series/` |
| `expec()` in objective-contributions | Yes (with compat auto-wrap shim) | No (implicit averaging) |
| Power operator `^` in expressions | No | Yes (right-associative, higher prec. than `*`/`/`) |
| `reduced_cost()` function | No | Yes (`extra-output`/`port-field-definitions` only) |
| `dual()` function | No | Yes (`extra-output`/`port-field-definitions` only) |
| `max`/`min`/`floor`/`ceil` with variable args rejected at parse time | No (fails at LP build) | Yes (explicit per-context forbidden-node check) |
| `model-decomposition` default resolution mode | `frontal` | `sequential-subproblems` |
| `models` list required in `optim-config.yml` | No | Yes |
