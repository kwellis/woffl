"""Jet Pump Network Solver

Add mutliple BatchPumps to a network and provide a shared resource. The shared
resource can be either lift water (power fluid) or total water.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ortools.algorithms.python import knapsack_solver

import woffl.assembly.curvefit as cf
from woffl.assembly.batchrun import BatchPump, validate_water
from woffl.geometry import JetPump


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


def optimize_jet_pumps(well_list: list[BatchPump], Qp_optm: np.ndarray, Qp_tot: float) -> pd.DataFrame:
    """Optimize Jet Pumps

    Run a discrete jet pump selection. Take the optimized power fluid rates from the continuous
    algorithm and use those as a starting point to pick the actual jet pumps. Each jet pump will
    have two different options that straddle the optimized power fluid rate, a high and low case.
    It is up to this algorithm to choose what is the best total outcome. It still needs to adhere
    to the total power fluid constraint, but the minimum and max individual constraints are no
    longer required.

    Args:
        well_list (list): List of the BatchPump Results
        Qp_optm (np.array): Array of the optimal power fluid rates to maximize oil
        Qp_tot (float): The total available surface pump capacity, bwpd

    Returns:
        df_kp (DataFrame): Need to figure this part out...
    """
    discrete_list = []

    for well, qpf_optm in zip(well_list, Qp_optm):
        df = well.df
        df["pf_diff"] = df["lift_wat"] - qpf_optm

        df_low = df[(df["semi"]) & (df["pf_diff"] < 0)]
        df_high = df[(df["semi"]) & (df["pf_diff"] >= 0)]

        if not df_low.empty and not df_high.empty:
            row_base = df_low.loc[df_low["pf_diff"].idxmax()]
            row_high = df_high.loc[df_high["pf_diff"].idxmin()]
            discrete_list.append(
                {
                    "wellname": well.wellname,
                    "qpf_optm": qpf_optm,
                    "qpf_base": row_base["lift_wat"],
                    "qpf_high": row_high["lift_wat"],
                    "qoil_base": row_base["qoil_std"],
                    "qoil_high": row_high["qoil_std"],
                    "jp_base": row_base["nozzle"] + row_base["throat"],
                    "jp_high": row_high["nozzle"] + row_high["throat"],
                    "qpf_diff": np.ceil(row_high["lift_wat"]) - np.floor(row_base["lift_wat"]),
                    "qoil_diff": np.ceil(row_high["qoil_std"]) - np.floor(row_base["qoil_std"]),
                }
            )
        else:
            # only one side exists — use closest semi-finalist as base, no high option
            row = df_low.loc[df_low["pf_diff"].idxmax()] if df_high.empty else df_high.loc[df_high["pf_diff"].idxmin()]
            discrete_list.append(
                {
                    "wellname": well.wellname,
                    "qpf_optm": qpf_optm,
                    "qpf_base": row["lift_wat"],
                    "qpf_high": np.nan,
                    "qoil_base": row["qoil_std"],
                    "qoil_high": np.nan,
                    "jp_base": row["nozzle"] + row["throat"],
                    "jp_high": np.nan,
                    "qpf_diff": 100,  # give it a false elevated weight so the solver won't pick it
                    "qoil_diff": -100,  # give it a false negative profit so solver doesn't select it
                }
            )

    # assume all base jetpumps to begin with. The decision is between keeping the base (0)
    # and going with a high jet pump (1). Bag space is the difference between the surface
    # pump available and base jet pump total powerfluid demand. Profit is the incremental oil
    # from base to high. Weight is the incremental power fluid from base to high.
    df_kp = pd.DataFrame(discrete_list)

    profit = [int(x) for x in df_kp["qoil_diff"]]
    weight = [int(x) for x in df_kp["qpf_diff"]]
    bag_size = int(np.ceil(Qp_tot - df_kp["qpf_base"].sum()))

    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, "JetPump_Knapsack"
    )

    solver.init(profit, [weight], [bag_size])
    solver.solve()
    df_kp["select_high"] = [
        solver.best_solution_contains(i) for i in range(len(profit))
    ]  # give true or false to round up

    return df_kp
