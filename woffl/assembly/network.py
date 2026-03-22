"""Jet Pump Network Solver

Add mutliple BatchPumps to a network and provide a shared resource. The shared
resource can be either lift water (power fluid) or total water.
"""

import numpy as np
import pandas as pd
from ortools.sat.python import cp_model

from woffl.assembly.batchrun import BatchPump

SCALE = 100  # CP-SAT requires integers; multiply floats by this before rounding


class WellNetwork:
    """A Well Network is a collection of multiple oil wells that share a common resource. The
    common resource is typically power fluid, but can also be total fluid"""

    def __init__(
        self,
        pwh_hdr: float | None,
        ppf_hdr: float | None,
        well_list: list[BatchPump],
        pad_name: str = "na",
    ) -> None:
        """Well Network Solver

        Used for feeding a network of multiple pumps together from a shared resource. The
        shared resource is typically power fluid, but can also be total fluid. If the well head
        header pressure or power fluid header pressure are left as None, then the well header
        pressure or power fluid pressure inside the individual BatchPumps will be used. This is
        convenient if a sensitivity is desired to look at changing power fluid pressure or well head

        Args:
            pwh_hdr (float): Pressure Wellhead Header, psig
            ppf_hdr (float): Pressure Power Fluid Header, psig
            batchlist (list): List of BatchPumps to run through
            pad_name (str): Name of the Pad or Network assessed
        """
        self.pwh_hdr = pwh_hdr
        self.ppf_hdr = ppf_hdr
        self.well_list = well_list
        self.pad_name = pad_name
        self._update_wells()
        self.results = False  # used for easily viewing if results have been processed

    def update_press(self, kind: str, psig: float) -> None:
        """Update Header Pressures

        Used to update different header pressure instead of re-initializing everything.
        Header pressures that can be updated include wellhead (production) or power fluid.

        Args:
            kind (str): Kind of Pressure to update. "wellhead" or "powerfluid".
            psig (float): Pressure to update with, psig
        """
        # chat gpt thinks using the reservoir method will fail, since you are calling ipr_su
        press_map = {"wellhead": "pwh_hdr", "powerfluid": "ppf_hdr"}

        # Validate the 'kind' argument
        if kind not in press_map:
            valid_kind = ", ".join(press_map.keys())
            raise ValueError(f"Invalid value for 'kind': {kind}. Expected {valid_kind}.")

        attr_name = press_map[kind]
        setattr(self, attr_name, psig)
        self._update_wells()

    def _update_wells(self) -> None:
        """Internal Method for Updating Well Pressures

        Can be run anytime a pressure is modified to update all of the wells in the list.
        Cascades network power fluid header pressure or well head header pressure.
        """
        if self.pwh_hdr is not None:
            for well in self.well_list:
                well.update_press("wellhead", self.pwh_hdr)

        if self.ppf_hdr is not None:
            for well in self.well_list:
                well.update_press("powerfluid", self.ppf_hdr)

    def add_well(self, well: BatchPump) -> None:
        """Add Well onto the Network"""
        self.well_list.append(well)
        self._update_wells()

    def drop_well(self, well: BatchPump) -> None:
        """Remove Well from the Network"""
        self.well_list.remove(well)

    def optimize(
        self,
        qpf_tot: float,
        water_key: str = "lift_wat",
        allow_shutin: bool = False,
    ) -> pd.DataFrame:
        """Run MCKP Optimization on Network Wells

        Args:
            qpf_tot (float): Total surface pump capacity, BWPD
            water_key (str): Column for capacity constraint, "lift_wat" or "totl_wat"
            allow_shutin (bool): If True, solver may shut in a well when its water is better used elsewhere

        Returns:
            df (DataFrame): One row per well with selected pump and rates
        """
        return optimize_jet_pumps(self.well_list, qpf_tot, water_key, allow_shutin)


def optimize_jet_pumps(
    well_list: list[BatchPump],
    qpf_tot: float,
    water_key: str = "lift_wat",
    allow_shutin: bool = False,
) -> pd.DataFrame:
    """Optimize Jet Pumps via Multiple-Choice Knapsack

    Each well picks exactly one jet pump from its semi-finalists to maximize
    total oil production subject to a shared power fluid capacity constraint.
    Uses the CP-SAT solver from ortools.

    Args:
        well_list (list[BatchPump]): Wells with batch_run() and process_results() already called
        qpf_tot (float): Total surface pump capacity, BWPD
        water_key (str): Column for capacity constraint, "lift_wat" or "totl_wat"
        allow_shutin (bool): If True, solver may shut in a well when its water is better used elsewhere

    Returns:
        df (DataFrame): One row per well with selected pump and rates

    Raises:
        ValueError: If any well has no semi-finalists
        RuntimeError: If the problem is infeasible (capacity too small)
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

    # capacity constraint: total water <= qpf_tot
    capacity_scaled = int(np.floor(qpf_tot * SCALE))
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
            f"MCKP infeasible: {qpf_tot:.0f} bwpd capacity cannot serve all wells. "
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
                    "nozzle": "off",
                    "throat": "off",
                    "qoil_std": 0.0,
                    "lift_wat": 0.0,
                    "form_wat": 0.0,
                    "totl_wat": 0.0,
                }
            )

    return pd.DataFrame(results)
