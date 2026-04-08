# Outputs: retrieving outputs with GemsPy


Once the optimisation problem was built and solved, one can retrieve the results as follows:

~~~ python
from gems.simulation.simulation_table import SimulationTableBuilder

df = SimulationTableBuilder().build(problem)
~~~

The result is a `pandas.DataFrame` with columns: `block`, `component`, `output`,
`absolute-time-index`, `block-time-index`, `scenario-index`, `value`, `basis-status`.

Reading the value of the optimisation variable `var_id` of component `component_id`
for a single time step and scenario reads:

~~~ python
value = df[(df["component"] == component_id) & (df["output"] == var_id)]["value"].iloc[0]
~~~

For multi-time or multi-scenario results, filter additionally by `block-time-index`
and `scenario-index`:

~~~ python
sub = df[(df["component"] == component_id) & (df["output"] == var_id)]
value_s0_t1 = sub[(sub["scenario-index"] == 0) & (sub["block-time-index"] == 1)]["value"].iloc[0]
~~~
