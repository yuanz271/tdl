# Examples

Place standalone runnable helpers and scripts in this directory. Execute them
directly from the project root, e.g. `python examples/<script>.py`, when you
need procedural utilities. The SARSA walkthrough ships as
`examples/sarsa.ipynb`; launch it via `jupyter lab examples/sarsa.ipynb`.
Keep imports relative to the installed package (`from tdl import sarsa`) so the
examples remain portable.

Install the optional tooling before opening the notebook:

- `uv sync --extra examples`
- `pip install -e .[examples]`
