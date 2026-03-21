# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style

Flag uncertainty — don't guess. If unsure about petroleum engineering concepts, correlation validity, physical assumptions, or whether something is correct, say so and ask. The user has domain expertise and will fill in the gaps. Getting it right together beats getting it wrong fast.

## Project

WOFFL (Water Optimization For Fluid Lift) — numerical solver for liquid-powered jet pumps with multiphase flow. Models subsurface oil well jet pump behavior and optimizes power fluid allocation across well networks.

Python >=3.10, AGPL-3.0 license, published to PyPI via GitHub Actions on tag push.

## Commands

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_flow_singlephase.py

# Run a specific test
pytest tests/test_flow_singlephase.py::TestClassName::test_method

# Formatting
black .
isort .

# Version bump
bumpver update --patch
```

## Testing

Run tests with `pytest tests/` after changes to flow, pvt, or assembly code. Three tiers:

- **Pure functions** (singlephase, pvt correlations): test against known reference values (e.g. Crane TP-410). Exact assertions with tight tolerances. These are low-maintenance and high-value.
- **Solvers/assembly** (jetpump_solver, batch_run, search_run): smoke tests with a reference well configuration. Assert outputs are physically reasonable (oil rate positive, pressures within bounds, convergence achieved) — use tolerances, not exact values.
- **Internals** (ptm_balance, throat momentum, secant iterations): don't test directly. If the top-level solve produces reasonable results, the internals are working. Testing these individually creates maintenance burden with little payoff.

Don't write tests for every function. Don't update tests unless the change actually affects expected behavior. Keep it lightweight.

## Architecture

The modeling pipeline flows top-down:

```
pvt (fluid properties) + flow.InFlow (IPR)
        ↓
geometry (JetPump, Pipe, WellProfile)
        ↓
flow (single/multiphase hydraulics, jet pump equations)
        ↓
assembly.BatchPump (single-well solver — iterates nozzle/throat combos)
        ↓
assembly.WellNetwork (multi-well power fluid allocation)
```

### Package breakdown

- **pvt/** — Fluid property classes: `BlackOil`, `FormGas`, `FormWater`, `ResMix`. Each has a `condition(press, temp)` method that caches thermodynamic properties at given P/T. Preset class methods (e.g. `BlackOil.schrader()`) provide canned field configurations.

- **geometry/** — `JetPump` (nozzle/throat catalogs from Champion X), `Pipe`/`PipeInPipe` (tubing/casing), `WellProfile` (MD/VD interpolation). Pipe presets: `four_half_tube()`, `three_half_tube()`, `seven_case()`. WellProfile presets: `schrader()`, `kuparuk()`.

- **flow/** — Core hydraulics. `singlephase.py` (friction, Reynolds), `twophase.py` (slip correlations), `jetflow.py` (nozzle/throat/diffuser energy balance), `outflow.py` (tubing pressure drop), `inflow.py` (Vogel IPR), `jetplot.py` (`JetBook` class for storing jet pump calculation arrays + visualization helpers), `jetgraphs.py` (multi-parameter graph generation). Test validation references Crane TP-410.

- **assembly/** — `BatchPump` ties everything together for a single well. Two modes:
  - `batch_run(jp_list)` — grid mode: iterates jet pump configurations, evaluates all combos.
  - `search_run(seed, lift_cost)` — search mode: uses Nelder-Mead to find the optimal continuous nozzle/throat diameters, then snaps to the nearest catalog pump. `lift_cost` (bbl oil / bbl lift water) penalizes power fluid usage; 0.0 = max oil, higher = favor smaller pumps.
  - Helper functions `continuous_jetpump()` (bypasses catalog lookup for arbitrary diameters) and `snap_to_catalog()` (finds nearest valid catalog pump by Euclidean distance).
  - `sysops.py` handles the physics: secant method (`qpf_secant`) for power fluid equilibrium. `curvefit.py` fits exponential models to batch results.

- **assembly/network.py** — `WellNetwork` class manages a collection of `BatchPump` wells sharing common header pressures (wellhead and power fluid). `optimize_jet_pumps(well_list, qpf_tot)` selects one jet pump per well via multiple-choice knapsack (ortools CP-SAT) to maximize total oil subject to shared power fluid capacity.

### Key solving patterns

- **Secant method** for power fluid equilibrium in `sysops.py`
- **Nelder-Mead** (scipy) for single-well continuous jet pump sizing in `batchrun.search_run`
- **Multiple-choice knapsack** (ortools CP-SAT) for multi-well discrete jet pump selection in `network.optimize_jet_pumps`

### Error handling

<!-- TODO: Has ConvergenceError been implemented? If so, remove this note and update the description below to reflect current state. -->

Inner solvers (jetflow, sysops) raise on convergence failure. Batch-level code (`batch_run`, `network_run`) uses a `debug` flag:
- `debug=False` (default): catch convergence errors, store in results as NaN, continue to next pump
- `debug=True`: re-raise so the traceback is visible for debugging a specific failure
- `search_run` optimizer objective always catches convergence errors and returns `1e10` penalty to let Nelder-Mead continue

**Planned**: Replace bare `Exception` catches with a custom `ConvergenceError(Exception)` class. Solvers raise `ConvergenceError` for iteration/convergence failures. `debug=False` catches only `ConvergenceError`. Input validation errors (`ValueError`) and real bugs (`TypeError`, `KeyError`) always propagate regardless of `debug` flag.

### Naming conventions

Variables follow a **prefix = quantity, suffix = location** pattern. Location suffixes track two converging flow paths through the jet pump:

```
Power fluid path:  _pf_surf (surface) → down annulus → _ni (nozzle inlet) → _nz (nozzle tip)
                                                                                    ↘
                                                                                _te (throat entry) → _tm (throat mix) → _di (diffuser) → _wh (wellhead)
                                                                                    ↗
Reservoir path:    _res (reservoir) → _su (suction/BHP) ─────────────────────────────
```

The pressure drop from `_res` to `_su` is the drawdown across the reservoir sand (IPR). Other suffixes: `_surf` (surface), `_std` (standard conditions), `_top`/`_bot`.

| Prefix | Quantity | Examples |
|--------|----------|----------|
| `p` | pressure | `psu`, `pte`, `ptm`, `pdi`, `pwh` |
| `t` | temperature | `tsu`, `tte`, `ttm`, `ttop` |
| `q` | volume flow | `qoil_std`, `qpf`, `qtot` |
| `m` | mass flow | `moil`, `mwat`, `mgas` |
| `rho_` | density | `rho_oil`, `rho_mix`, `rho_nz` |
| `u` | viscosity | `uoil`, `uwat`, `umix`, `uod` |
| `v` | velocity | `vnz`, `vte`, `vtm`, `vsl` |
| `a` | area | `anz`, `ate`, `ath`, `adi` |
| `d` | diameter | `dnz`, `dth`, `dhyd` |
| `k` | loss coefficient | `knz`, `ken`, `kth`, `kdi` |
| `x` | mass fraction | `xoil`, `xwat`, `xgas` |
| `y` | volume fraction | `yoil`, `ywat`, `ygas` |
| `N` | dimensionless number | `NRe`, `NFr`, `NLv` |

Other conventions:
- `prop_` prefix for ResMix objects: `prop_su`, `prop_pf`, `prop_tm`
- `ff` = Darcy friction factor, `wc` = watercut, `fgor` = formation GOR, `rs` = solution GOR
- `_ray` suffix = numpy array, `_list` = Python list, `_df` = DataFrame, `_book` = JetBook
- Density uses `rho_` prefix (not `d`)

## Docstring Style

All new and modified code must use **Google-style docstrings** following this pattern:

```python
def func_name(press: float, temp: float, fluid: ResMix) -> tuple[float, float]:
    """Short Descriptive Title

    Optional expanded description with physics context, assumptions,
    or implementation notes.

    Args:
        press (float): Pressure, psig
        temp (float): Temperature, deg F
        fluid (ResMix): Reservoir mixture at suction conditions

    Returns:
        vel (float): Velocity, ft/s
        rho (float): Density, lbm/ft3

    """
```

Rules:
- **Title line**: short name, no trailing period
- **Args/Returns**: `name (type): Description, units` — always include units for dimensional quantities
- **Returns with tuples**: one line per element
- **References**: do NOT add References sections — risk of hallucinating paper titles, SPE codes, or authors is too high. Only preserve existing references already in the codebase
- **Raises**: include when a function raises exceptions (convergence failures, invalid inputs, etc.)
- **Notes**: include for validity bounds and assumptions — e.g. "valid above bubblepoint only", "assumes turbulent flow (Re > 4000)". If unsure of the exact validity boundaries for a correlation or equation, flag it with `# TODO: confirm validity range` and ask the user rather than guessing
- **Properties**: single-line with units — `"""Jet Pump TVD, Feet"""`
- **Classes**: brief summary on class; detail in `__init__`
- **Modules**: high-level purpose paragraph, no Args/Returns
- **Units**: all oilfield/US customary. Some correlations convert to metric internally — always verify a function's actual unit expectations before documenting or calling it.
  - Pressure: `psig` (gauge), `psia` (absolute = gauge + 14.7), `psi⁻¹` (compressibility)
  - Temperature: `deg F` (normal), `deg R` (absolute = F + 459.67)
  - Large distances (well depths, MD/VD): `feet`
  - Small distances (nozzle, throat, pipe diameters, roughness): `inches`
  - Area: `ft2` (converted from inches via ÷144)
  - Flow rates: `BPD` / `BWPD` / `BOPD` / `STBOPD` (converted to `ft3/s` internally)
  - Gas solubility: `SCF/STB`; FVF: `rb/stb`
  - Density: `lbm/ft3`; Viscosity: `cP` (÷1488.2 for `lbm/(ft·s)` in Reynolds)
  - Velocity: `ft/s`; Surface tension: `lbf/ft` (correlations use dynes/cm internally)
  - Specific gravity, Reynolds, friction factor, holdup, watercut: `dimensionless`

## Examples

In `examples/`: `e41_singlepump.py`, `e41_batchpump.py` (grid search), `e41_searchpump.py` (Nelder-Mead search with lift_cost sweep), `e41_direction.py`, `epad_mckp.py` (multi-well MCKP network optimization), `flow_singlephase.py`/`flow_multiphase.py` (hydraulics demos).
