"""Jet Pump Network Solver

Add mutliple BatchPumps to a network and provide a shared resource. The shared
resource can be either lift water (power fluid) or total water.
"""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

    def network_run(self, jetpumps: list[JetPump], debug: bool = False) -> None:
        """Network Run of Wells

        Run through multiple wells with different types of jet pumps. Results are
        stored as dataframes on each BatchPump and can be viewed later. Results
        are processed for creating master curve and future plotting.

        Args:
            jetpumps (list): List of JetPumps
            debug (bool): True - Errors are Raised, False - Errors are Stored
        """
        for well in self.well_list:
            well.batch_run(jetpumps, debug)
            well.process_results()
        self.results = True  # tracker to know if results have been ran

    def master_curves(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Create the Network Master Curves

        Creates two master curves that are used for optimizing jet pumps selection
        on the network selection.

        Args:
            None

        Returns:
            mowr_ray (np.ndarray): Marginal Oil Water Ratio
            lwat_pad (np.ndarray): Lift Water of the Pad, BWPD
            twat_pad (np.ndarray): Total Water of the Pad, BWPD
        """
        if self.results is False:
            raise ValueError("Run network before generating master curves")

        mowr_ray = np.arange(0.01, 1.1, 0.01)
        lwat_pad = np.zeros_like(mowr_ray)
        twat_pad = np.zeros_like(mowr_ray)

        loil_pad = np.zeros_like(mowr_ray)  # oil rate predicted using lift water coeff
        toil_pad = np.zeros_like(mowr_ray)  # oil rate predicted using total water coeff

        for well in self.well_list:

            qoil_lift, lwat_well = well.theory_curves(mowr_ray, "lift")
            qoil_totl, twat_well = well.theory_curves(mowr_ray, "total")

            lwat_pad = lwat_pad + lwat_well  # add numpy arrays element by element
            twat_pad = twat_pad + twat_well  # add numpy arrays element by element

            loil_pad = loil_pad + qoil_lift
            toil_pad = toil_pad + qoil_totl

        self.mowr_ray = mowr_ray
        self.lwat_pad = lwat_pad  # rename these?
        self.twat_pad = twat_pad  # rename these?

        self.loil_pad = loil_pad
        self.toil_pad = toil_pad

        return mowr_ray, lwat_pad, twat_pad

    def equal_slope(self, qwat_pad: float, water: str) -> float:
        """Calculate Equal Slope of Wells on Network

        Provide the shared water resource of all the wells that are on the
        same network together. Method will calculate the approximate slope
        that all the wells should be operating at to evenly distribute water.

        Args:
            qwat_pad (float): Flow of Water for the Pad / Network, BPD
            water (str): "lift" or "total" depending on the desired analysis

        Returns:
            mowr (float): Target Marginal Oil Water Rate for Wells to Operate
        """
        water = validate_water(water)
        self.master_curves()

        # Determine the correct water data and coefficients
        qwat_ray = self.lwat_pad if water == "lift" else self.twat_pad
        mowr = np.interp(qwat_pad, np.flip(qwat_ray), np.flip(self.mowr_ray))  # not sorted from largest to smallest...
        return float(mowr)

    def dist_slope(self, mowr_pad: float, water: str) -> pd.DataFrame:
        """Distribute the Slope Equally

        Provide a target mowr and constraint on the wells. The method will
        go through and select all the wells. (Should this just be added to
        the other method? equal_slope). Ideally you would total up all the
        wells and look for additional capacity, then go bump that well up.

        Args:
            mowr_pad (float):
            water (str): "lift" or "total" depending on the desired analysis
        """
        water = validate_water(water)
        attr_name = "coeff_lift" if water == "lift" else "coeff_totl"
        col_name = "lift_wat" if water == "lift" else "totl_wat"

        result_list = []

        for well in self.well_list:
            a, b, c = getattr(well, attr_name)  # pull out exponetial coefficients
            qwat_tgt = cf.rev_exp_deriv(mowr_pad, b, c)  # target water rate

            # semi is true and the water rate is less than the target rate, is there a way to say
            # the datapoint that is closer instead of just the one that is less? Will this mess you up?
            df_semi = well.df[(well.df["semi"] == True) & (well.df[col_name] < qwat_tgt)]  # noqa: E712
            idx_jp = df_semi[col_name].idmax()  # index the desired jetpump is at
            row_jp = well.df[idx_jp]
            row_jp["wellname"] = well.wellname

            result_list.append(row_jp)

        df_net = pd.DataFrame(result_list)
        self.df = df_net
        return df_net

    def network_plot_data(self, water: str, curve: bool = False) -> None:
        """Plot Data

        Plot an array to visualize the performance of all the wells that are
        on the prescribed network.

        Args:
            water (str): "lift" or "total" depending on the desired x axis
            curve (bool): Show the curve fit or not
        """
        water = validate_water(water)
        n_wells = len(self.well_list)  # how many wells are there
        n_cols = 4
        n_rows = 1 + ((n_wells - 1) // n_cols)  # integer division

        fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))

        axs = axs.flatten() if n_wells > 1 else [axs]  # type: ignore
        for well, ax in zip(self.well_list, axs):
            well.plot_data(water, curve, ax)  # type: ignore

        # hide the extra subplots
        for i in range(len(self.well_list), len(axs)):
            axs[i].axis("off")  # type: ignore

        plt.show()

    def network_plot_derv(self, water: str) -> None:
        """Plot Derivative Marginal

        Plot the various wells marginal oil water ratio on one graph
        to show how the various curves would line up if trying to match
        the mwor value.

        Args:
            water (str): "lift" or "total" depending on the desired x axis
        """
        water = validate_water(water)

        # add a horizontal line where the pad mwor was calculated to be at

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab20", len(self.well_list))  # generate list of colors to plot

        for i, well in enumerate(self.well_list):
            well._plot_derv_network(water, ax, mcolor=cmap(i))  # type: ignore

        ax.set_xlabel(f"{water.capitalize()} Water Rate, BWPD")
        ax.set_ylabel("Marginal Oil Water Rate, Oil BBL / Water BBL")
        ax.title.set_text("Marginal Network Jet Pump Performance")
        ax.legend()

        plt.show()

    def network_plot_master(self, water: str) -> None:
        """Plot Network Master Curves

        Plot the master curve that is produced by adding up.

        Args:
            water (str): "lift" or "total" depending on the desired x axis
        """
        water = validate_water(water)
        self.master_curves()
        # Determine the correct water data and coefficients
        qwat_ray = self.lwat_pad if water == "lift" else self.twat_pad
        qoil_ray = self.loil_pad if water == "lift" else self.toil_pad

        fig, ax = plt.subplots()
        ax2 = ax.twinx()

        pri = ax.plot(qwat_ray, self.mowr_ray, color="blue", linestyle="--", label="marginal")
        sec = ax2.plot(qwat_ray, qoil_ray, color="red", linestyle="--", label="oil")  # type: ignore

        leg_nms = pri + sec
        labs = [leg.get_label() for leg in leg_nms]
        ax.legend(leg_nms, labs, loc="center right")  # type: ignore

        ax.set_xlabel(f"Network Required {water.capitalize()} Water, BWPD")
        ax.set_ylabel(f"Network Marginal Oil {water.capitalize()} Water Ratio, bbl oil / bbl water")

        ax2.set_ylabel("Oil Rate, BOPD")

        plt.title(f"{self.pad_name} Pad Master {water.capitalize()} Water Curve")
        plt.show()
