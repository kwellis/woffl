"""Multiple-Choice Knapsack for Jet Pump Network Optimization

Selects exactly one jet pump per well from each well's semi-finalists to
maximize total oil production subject to a shared power fluid capacity.

Replaces the two-phase approach (continuous reduced Newton + 0/1 knapsack)
with a single discrete optimization over actual BatchPump simulation results.

Solver: ortools CP-SAT
"""

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from woffl.assembly.batchrun import BatchPump

SCALE = 100  # CP-SAT requires integers; multiply floats by this before rounding


def optimize_mckp(
    well_list: list[BatchPump],
    Qp_tot: float,
    water_key: str = "lift_wat",
    allow_shutin: bool = False,
) -> pd.DataFrame:
    """Multiple-Choice Knapsack for Jet Pump Selection

    Each well picks exactly one jet pump from its semi-finalists to maximize
    total oil production subject to a shared power fluid capacity constraint.

    Args:
        well_list: BatchPump objects with batch_run() and process_results() already called.
        Qp_tot: Total available power fluid capacity, bwpd.
        water_key: DataFrame column for the capacity constraint. Default "lift_wat"
            for power fluid only; use "totl_wat" to constrain total water.
        allow_shutin: If True, the solver may leave a well without a pump (0 oil,
            0 water) when its power fluid is better used elsewhere.

    Returns:
        DataFrame with one row per well showing the selected jet pump and rates.
        Columns: wellname, nozzle, throat, qoil_std, lift_wat, form_wat, totl_wat.
        Shutin wells have "shutin" for nozzle/throat and 0.0 for all rates.

    Raises:
        ValueError: If any well has no semi-finalists.
        RuntimeError: If the problem is infeasible (capacity too small).
    """
    # collect semi-finalist candidates per well
    candidates = []
    for well in well_list:
        df_semi = well.df[well.df["semi"]].reset_index(drop=True)
        if df_semi.empty:
            raise ValueError(f"Well '{well.wellname}' has no semi-finalists")
        candidates.append(df_semi)

    model = cp_model.CpModel()

    # decision variables: x[i][j] = 1 if well i selects semi-finalist j
    x = []
    for i, df_semi in enumerate(candidates):
        well_vars = [model.new_bool_var(f"w{i}_p{j}") for j in range(len(df_semi))]
        x.append(well_vars)
        if allow_shutin:
            model.add_at_most_one(well_vars)
        else:
            model.add_exactly_one(well_vars)

    # capacity constraint: total water <= Qp_tot
    capacity_scaled = int(np.floor(Qp_tot * SCALE))
    water_terms = []
    for i, df_semi in enumerate(candidates):
        for j, wat in enumerate(df_semi[water_key]):
            water_terms.append(int(np.ceil(wat * SCALE)) * x[i][j])
    model.add(sum(water_terms) <= capacity_scaled)

    # objective: maximize total oil
    oil_terms = []
    for i, df_semi in enumerate(candidates):
        for j, oil in enumerate(df_semi["qoil_std"]):
            oil_terms.append(int(np.floor(oil * SCALE)) * x[i][j])
    model.maximize(sum(oil_terms))

    # solve
    solver = cp_model.CpSolver()
    status = solver.solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        min_water = sum(df[water_key].min() for df in candidates)
        raise RuntimeError(
            f"MCKP infeasible: {Qp_tot:.0f} bwpd capacity cannot serve all wells. "
            f"Minimum required: {min_water:.0f} bwpd."
        )

    # extract solution
    results = []
    for i, (well, df_semi) in enumerate(zip(well_list, candidates)):
        selected = False
        for j in range(len(df_semi)):
            if solver.value(x[i][j]):
                row = df_semi.iloc[j]
                results.append(
                    {
                        "wellname": well.wellname,
                        "nozzle": row["nozzle"],
                        "throat": row["throat"],
                        "qoil_std": row["qoil_std"],
                        "lift_wat": row["lift_wat"],
                        "form_wat": row["form_wat"],
                        "totl_wat": row["totl_wat"],
                    }
                )
                selected = True
                break
        if not selected:
            results.append(
                {
                    "wellname": well.wellname,
                    "nozzle": "shutin",
                    "throat": "shutin",
                    "qoil_std": 0.0,
                    "lift_wat": 0.0,
                    "form_wat": 0.0,
                    "totl_wat": 0.0,
                }
            )

    return pd.DataFrame(results)
