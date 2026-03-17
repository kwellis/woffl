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
