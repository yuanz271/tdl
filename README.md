# Temporal Difference Learning (TDL)

Reinforcement learning experiments exploring SARSA-based policy fitting for behavioural datasets.

## Installation

Install dependencies into a Python 3.11 environment:

```bash
uv sync
```

Include notebook tooling such as JupyterLab:

```bash
uv sync --extra examples
```

Alternatively, rely on an editable install:

```bash
pip install -e .
```

For extras:

```bash
pip install -e .[examples]
```

## Usage

Explore the default learning session against `examples/M1.csv`:

```bash
jupyter lab examples/sarsa.ipynb
```
