# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

**Install:**
```bash
pip install -r requirements.txt -r requirements-dev.txt
```

**Test:**
```bash
pytest                                          # all tests
pytest tests/path/to/test_file.py::test_name   # single test
pytest --cov gems --cov-report xml             # with coverage
```

**Lint & Format:**
```bash
black --config pyproject.toml src/ tests/
isort --profile black --filter-files src/ tests/
mypy
```

**CLI entry point:**
```bash
gemspy \
  --model-libs  path/to/lib1.yml path/to/lib2.yml \
  --components   path/to/components.yml \
  --timeseries   path/to/timeseries/ \
  --duration     8760 \
  --scenarios    1
```

See also `AGENTS.md` for an architecture overview and `docs/agents/` for deeper guidance.

## Architecture

The pipeline flows: **YAML → parsing → model resolution → network building → optimization problem → HiGHS solver → results**

### Core Modules (`src/gems/`)

**`expression/`** — Immutable AST for the mathematical expression language.
- All nodes are `@dataclass(frozen=True)` subclasses of `ExpressionNode`.
- Operators: arithmetic, comparisons, time/scenario (`time_sum`, `time_shift`), functions (`max`, `min`, `ceil`, `floor`), and port field references.
- Grammar source is `grammar/Expr.g4`; generated ANTLR4 parser lives in `expression/parsing/antlr/` — **do not edit those files directly**, regenerate from the grammar.
- All tree operations use `ExpressionVisitor` subclasses (evaluation, linearization, degree analysis, printing). Add new operations by subclassing `ExpressionVisitor`, not by branching on node types.

**`model/`** — Immutable model templates loaded from YAML.
- `Model` → parameters, variables, constraints, ports. Never instantiated directly.
- `Library` → collection of `Model` objects.
- YAML parsing uses Pydantic v2 with alias generators (kebab-case keys → PascalCase/camelCase fields). Use `ConfigDict(alias_generator=...)` consistently.

**`study/`** — Study definition and network instantiation.
- `System` (parsed) → `Network` (instantiated graph of `Node`s, `Component`s, connections).
- `Component` = a `Model` instance with concrete parameter values.
- `DataBase` manages time series (`xarray.DataArray`) and scenario data.

**`simulation/`** — Optimization problem construction and solving.
- `LinopyOptimizationProblem.build_problem()` runs four sequential phases:
  1. **Parameter arrays** — `_build_param_arrays_for_model()`: converts DB values to `xarray.DataArray` indexed by `[component, time, scenario]`.
  2. **Decision variables** — one linopy `Variable` per model variable, over all components.
  3. **Port arrays** — resolves port connections via incidence matrix.
  4. **Constraints & objective** — AST traversal via `VectorizedLinopyBuilder` (in `linopy_linearize.py`).
- `BendersDecomposedProblem` handles temporal decomposition for large problems.
- `OutputValues` extracts and formats solver results.

**`main/`** — CLI orchestration; `main.py` is the entry point wiring all phases together.

**`libs/`** — Built-in standard model library (generators, storage, demand, etc.) shipped as YAML.

### Key Design Patterns

- **Frozen dataclasses** everywhere for value objects (expression nodes, model definitions, constraints).
- **Visitor pattern** for all expression tree operations — subclass `ExpressionVisitor`, never switch on node type.
- **Explicit indexing**: parameters and variables carry time and scenario dimensions; expressions track these automatically via `IndexingVisitor`.
- **`ValueType`** enum (`INTEGER`, `CONTINUOUS`, `BOOLEAN`) for variable typing.
- **Pydantic v2** for YAML round-tripping; use v2 APIs only (no `validator`/`root_validator`).
- **No solver mocking in tests**: integration and unit tests use real HiGHS calls.

## Conventions

- **Type annotations**: all functions must be fully typed. mypy runs in strict mode (`disallow_untyped_defs`, `disallow_untyped_calls`).
- **Naming**: `PascalCase` for classes, `snake_case` for functions/variables, `UPPER_SNAKE_CASE` for constants, `kebab-case` for YAML keys.
- **Commit messages**: Conventional Commits — `feat(scope): summary`, `fix(scope): summary`, etc.
- **Do not edit** files under `src/gems/expression/parsing/antlr/` — regenerate from `grammar/Expr.g4`.
- **Do not bump** the version in `pyproject.toml` manually.