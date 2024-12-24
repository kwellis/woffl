"""Batch Jet Pump Runs

Contains code that is used to run multiple pumps at once to understand the
current conditions. Code outputs a formatted list of python dictionaries that
can be converted to a Pandas Dataframe or equivalent for analysis.
"""

import os
from dataclasses import dataclass
from itertools import product

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

import woffl.assembly.curvefit as cf
import woffl.assembly.sysops as so
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Annulus, Pipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.resmix import ResMix


@dataclass(frozen=True)
class BatchResult:
    """Code is currently not used"""

    wellname: str
    nozzle: str
    throat: str
    choked: bool
    qoil_std: float
    fwat_bwpd: float
    mach_te: float
    total_water: float
    total_wc: float


class BatchPump:
    "Used for running different nozzle and throat combinations and selecting the correct one."

    def __init__(
        self,
        pwh: float,
        tsu: float,
        rho_pf: float,
        ppf_surf: float,
        wellbore: Pipe,
        wellprof: WellProfile,
        ipr_su: InFlow,
        prop_su: ResMix,
        wellname: str = "na",
    ) -> None:
        """Batch Pump Solver

        Used for iterating across a wide range of different pumps. An adjacent dataclass will
        be made for storing the results. Could add an optional wellname here if desired?

        Args:
            pwh (float): Pressure Wellhead, psig
            tsu (float): Temperature Suction, deg F
            rho_pf (float): Density of the power fluid, lbm/ft3
            ppf_surf (float): Pressure Power Fluid Surface, psig
            wellbore (Pipe): Pipe Class of the Wellbore
            wellprof (WellProfile): Well Profile Class
            ipr_su (InFlow): Inflow Performance Class
            prop_su (ResMix): Reservoir Mixture Conditions
        """
        self.pwh = pwh
        self.tsu = tsu
        self.rho_pf = rho_pf
        self.ppf_surf = ppf_surf
        self.wellbore = wellbore
        self.wellprof = wellprof
        self.ipr_su = ipr_su
        self.prop_su = prop_su
        self.wellname = wellname

    def update_press(self, kind: str, psig: float) -> None:
        """Update Pressure

        Used to update different pressures instead of re-initializing everything. Components that
        can be updated include the well head, the power fluid or the reservoir pressure.

        Args:
            kind (str): Kind of Pressure to update. Either "wellhead", "powerfluid" or "reservoir"
            psig (float): Pressure to update with, psig
        """
        # chat gpt thinks using the reservoir method will fail, since you are calling ipr_su
        press_map = {"wellhead": "pwh", "powerfluid": "ppf_surf", "reservoir": "ipr_su.pres"}

        # Validate the 'kind' argument
        if kind not in press_map:
            valid_kind = ", ".join(press_map.keys())
            raise ValueError(f"Invalid value for 'kind': {kind}. Expected {valid_kind}.")

        attr_name = press_map[kind]
        setattr(self, attr_name, psig)

    @staticmethod
    def jetpump_list(
        nozzles: list[str],
        throats: list[str],
        knz: float = 0.01,
        ken: float = 0.03,
        kth: float = 0.3,
        kdi: float = 0.4,
    ) -> list[JetPump]:
        """Create a list of Jet Pumps

        Automatically generate a list of different types of jet pumps. The list is fed into
        the batch run to understand performance. Could move this into batch run if desired.

        Args:
            nozzles (list): List of Nozzle Strings
            throats (list): List of Throat Ratios
            knz (float): Nozzle Friction Factor
            ken (float): Enterance Friction Factor
            kth (float): Throat Friction Factor
            kdi (float): Diffuser Friction Factor

        Returns:
            jp_list (list): List of JetPumps
        """
        jp_list = []
        for nozzle, throat in product(nozzles, throats):
            jp_list.append(JetPump(nozzle, throat, knz, ken, kth, kdi))
        return jp_list

    def batch_run(self, jetpumps: list[JetPump], debug: bool = False) -> pd.DataFrame:
        """Batch Run of Jet Pumps

        Run through multiple different types of jet pumps. Results will be stored in
        a dataframe where the results can be graphed and selected for the optimal pump.
        The dataframe is added to the class as a variable for future inspection.

        Args:
            jetpumps (list): List of JetPumps
            debug (bool): True - Errors are Raised, False - Errors are Stored

        Returns:
            df (DataFrame): DataFrame of Jet Pump Results
        """
        results = []
        for jetpump in jetpumps:
            try:
                psu_solv, sonic_status, qoil_std, fwat_bpd, lwat_bpd, mach_te = so.jetpump_solver(
                    self.pwh,
                    self.tsu,
                    self.rho_pf,
                    self.ppf_surf,
                    jetpump,
                    self.wellbore,
                    self.wellprof,
                    self.ipr_su,
                    self.prop_su,
                )
                result = {
                    "nozzle": jetpump.noz_no,
                    "throat": jetpump.rat_ar,
                    "sonic_status": sonic_status,
                    "mach_te": mach_te,
                    "psu_solv": psu_solv,
                    "qoil_std": qoil_std,
                    "form_wat": fwat_bpd,
                    "lift_wat": lwat_bpd,
                    "totl_wat": fwat_bpd + lwat_bpd,
                    "form_wor": fwat_bpd / qoil_std,
                    "totl_wor": (fwat_bpd + lwat_bpd) / qoil_std,
                    "error": "na",
                }

            except Exception as exc:
                if debug:
                    print(f"Failed on nozzle {jetpump.noz_no} and throat: {jetpump.rat_ar}")
                    raise exc
                else:
                    result = {
                        "nozzle": jetpump.noz_no,
                        "throat": jetpump.rat_ar,
                        "sonic_status": False,
                        "mach_te": np.nan,
                        "psu_solv": np.nan,
                        "qoil_std": np.nan,
                        "form_wat": np.nan,
                        "lift_wat": np.nan,
                        "totl_wat": np.nan,
                        "form_wor": np.nan,
                        "totl_wor": np.nan,
                        "error": exc,
                    }
            results.append(result)  # add some progress bar code here?
        self.df = pd.DataFrame(results)
        return self.df  # should this be returned as none?

    def process_results(self) -> pd.DataFrame:
        """Process Results

        Identify the semi-finalist jet pumps, calculate numerical gradients and curve fits.
        The semi-fialist jet pump means no other jet pump makes more oil for less water.
        The gradients are MOTWR, stands for marginal oil total water ratio and MOLWR, stands
        for marginal oil lift water ratio. Calculate curve fit coefficients for the jet
        pump lift and total water vs oil.

        Args:
            self

        Returns:
            df (DataFrame): Adds semi (bool) column, motwr, molwr.
        """
        semi_mask = batch_results_mask(self.df["qoil_std"], self.df["totl_wat"], self.df["nozzle"])
        self.df["semi"] = semi_mask

        semi_df = self.df[self.df["semi"]].copy()
        if semi_df.empty:
            raise ValueError(f"No Semi-Finalist Jet Pumps on {self.wellname}")

        semi_df = semi_df.sort_values(by="qoil_std", ascending=True)

        qoil_semi = semi_df["qoil_std"].to_numpy()
        twat_semi = semi_df["totl_wat"].to_numpy()
        lwat_semi = semi_df["lift_wat"].to_numpy()

        semi_df["motwr"] = gradient_back(qoil_semi, twat_semi)
        semi_df["molwr"] = gradient_back(qoil_semi, lwat_semi)

        self.df = self.df.merge(semi_df[["motwr", "molwr"]], left_index=True, right_index=True, how="left")

        self.coeff_totl = cf.batch_curve_fit(qoil_semi, twat_semi, origin=False)
        self.coeff_lift = cf.batch_curve_fit(qoil_semi, lwat_semi, origin=False)

        return self.df

    def theory_curves(self, mowr_ray: np.ndarray, water: str) -> tuple[np.ndarray, np.ndarray]:
        """Theoretical Performace Curves

        Create theoretical jet pump performance curves using marginal oil water rate
        as the input. Generate arrays of the theoretical oil rat and water rate.

        Args:
            mowr_ray (np.ndarry): Marginal Oil Water Ratio, bbl/bbl
            water (str): "lift" or "total" depending on the desired analysis

        Returns:
            qoil_std (np.ndarray): Oil Rate at different MOWR values
            qwat_bpd (np.ndarray): Water Rate at different MOWR values
        """
        water = validate_water(water)
        coeff = self.coeff_lift if water == "lift" else self.coeff_totl
        a, b, c = coeff

        qwat_bpd = np.array([cf.rev_exp_deriv(mowr, b, c) for mowr in mowr_ray])
        qoil_std = np.array([cf.exp_model(qwat, a, b, c) for qwat in qwat_bpd])

        return qoil_std, qwat_bpd

    def plot_data(
        self, water: str, curve: bool = False, ax: Axes | None = None, fig_path: str | os.PathLike | None = None
    ) -> None:
        """Plot Data

        Plot the results from the jet pump batch run to visualize the performance
        of each jet pump. The figure path allows the figure to be locally saved if passed.

        Args:
            water (str): "lift" or "total" depending on the desired x axis
            curve (bool): Show the curve fit or not
            ax (Axes): Matplotlib axes
            fig_path (Path): Path to optional output file, saves files to be viewed later
        """
        water = validate_water(water)

        # Determine the correct water data and coefficients
        qwat_bpd = self.df["lift_wat"] if water == "lift" else self.df["totl_wat"]
        coeff = self.coeff_lift if water == "lift" else self.coeff_totl  # type: ignore
        coeff = coeff if curve else None

        hold = 1  # define the variable with anything, need to trade
        if ax is None:
            hold = None  # transfer None value to something not used
            fig, ax = plt.subplots(figsize=(5.5, 4))

        # calling the function to plot the data
        batch_plot_data(
            qoil_std=self.df["qoil_std"],
            qwat_bpd=qwat_bpd,
            water=water,
            nozzles=self.df["nozzle"],
            throats=self.df["throat"],
            semi=self.df["semi"],
            wellname=self.wellname,
            coeff=coeff,
            ax=ax,  # type: ignore
        )

        if hold is None:
            if fig_path is not None:
                plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            else:
                plt.show()

    def plot_derv(self, water: str, fig_path: str | os.PathLike | None = None) -> None:
        """Plot Derivative

        Plot the derivative results from the jet pump batch run to visualize how
        well of a match occured between the data and curve for each jet pump

        Args:
            water (str): "lift" or "total" depending on the desired x axis
            fig_path (Path): Path to optional output file, saves files to be viewed later
        """
        # Validate the 'water' argument
        water = validate_water(water)
        df_semi = self.df[self.df["semi"]]  # filter out the bad parts

        # Determine the correct water data and coefficients
        marg_semi = df_semi["molwr"] if water == "lift" else df_semi["motwr"]
        qwat_semi = df_semi["lift_wat"] if water == "lift" else df_semi["totl_wat"]
        coeff = self.coeff_lift if water == "lift" else self.coeff_totl

        fig, ax = plt.subplots(figsize=(5.5, 4))

        # calling the function to plot the derivatives
        batch_plot_derv_base(
            marg_filt=marg_semi,
            qwat_filt=qwat_semi,
            nozz_filt=df_semi["nozzle"],
            thrt_filt=df_semi["throat"],
            wellname=self.wellname,
            coeff=coeff,
            ax=ax,
            mcolor="blue",
            network=False,
        )

        ax.set_xlabel(f"{water.capitalize()} Water Rate, BWPD")
        ax.set_ylabel("Marginal Oil Water Rate, Oil BBL / Water BBL")
        ax.title.set_text(f"{self.wellname} Jet Pump Performance")
        ax.legend()

        if fig_path is not None:
            plt.savefig(fig_path, dpi=300)
        else:
            plt.show()

    def _plot_derv_network(self, water: str, ax: Axes, mcolor: str | mcolors.Colormap) -> None:
        """Plot Derivative in a Network

        Plot the derivative results from the jet pump batch run to visualize how
        well of a match occured between the data and curve for each jet pump

        Args:
            water (str): "lift" or "total" depending on the desired x axis
            curve (bool): Show the curve fit or not
            ax (Axes): Matplotlib axes
            mcolor (str): Matplotlib color
        """
        # Validate the 'water' argument
        water = validate_water(water)
        df_semi = self.df[self.df["semi"]]  # filter out the bad parts

        # Determine the correct water data and coefficients
        marg_semi = df_semi["molwr"] if water == "lift" else df_semi["motwr"]
        qwat_semi = df_semi["lift_wat"] if water == "lift" else df_semi["totl_wat"]
        coeff = self.coeff_lift if water == "lift" else self.coeff_totl

        # calling the function to plot the derivatives
        batch_plot_derv_base(
            marg_filt=marg_semi,
            qwat_filt=qwat_semi,
            nozz_filt=df_semi["nozzle"],
            thrt_filt=df_semi["throat"],
            wellname=self.wellname,
            coeff=coeff,
            ax=ax,
            mcolor=mcolor,
            network=True,
        )

        return None


def validate_water(water: str) -> str:
    """Validate Type of Water String

    Checks that the string passed into a method or arguement fits the required description.
    This is used when the water type wants to be defined as lift or total

    Args:
        water (str): "lift" or "total" depending on the desired x axis

    Returns:
        water (str): Properly formatted as either "lift" or "total"
    """
    # Validate the 'water' argument
    if water not in {"lift", "total", "totl"}:
        raise ValueError(f"Invalid value for 'water': {water}. Expected 'lift', 'total', or 'totl'.")

    # Standardize "totl" to "total"
    if water == "totl":
        water = "total"
    return water


def batch_results_mask(
    qoil_std: np.ndarray | pd.Series,
    qwat_tot: np.ndarray | pd.Series,
    nozzles: np.ndarray | pd.Series,
) -> np.ndarray:
    """Batch Results Mask

    Create a mask of booleans from batch results. The initial filter is selecting the throat with the highest
    oil rate for each nozzle size. Any points where the oil rate is lower for a higher amount of
    water are then next removed. The filtered points can be passed to a function to calculate the gradient.

    Args:
        qoil_std (np.ndarray | pd.Series): Oil Prod. Rate, BOPD
        qwat_tot (np.ndarray | pd.Series): Total Water Rate, BWPD
        nozzles (np.ndarray | pd.Series): List of the Nozzles

    Returns:
        np.ndarray: True is a point to calc gradient, False means a point exists with better oil and less water
    """
    # convert into numpy arrays
    qoil_std = np.asarray(qoil_std)
    qwat_tot = np.asarray(qwat_tot)
    nozzles = np.asarray(nozzles)

    mask = np.zeros(len(qoil_std), dtype=bool)

    # start by picking the highest oil rate for each nozzle
    unique_nozzles = np.unique(nozzles)  # unique nozzles in the list
    for noz in unique_nozzles:
        noz_idxs = np.where(nozzles == noz)[0]  # indicies where the nozzle is a specific nozzle

        if np.all(np.isnan(qoil_std[noz_idxs])):  # skip the nozzle if they are all nan
            continue

        best_idx = noz_idxs[np.nanargmax(qoil_std[noz_idxs])]
        mask[best_idx] = True

    # compare the points to themselves to look for where oil is higher for less water
    for idx in np.where(mask)[0]:
        higher_wat_mask = qwat_tot < qwat_tot[idx]
        lower_oil_mask = qoil_std > qoil_std[idx]
        if np.any(higher_wat_mask & lower_oil_mask):
            mask[idx] = False

    return mask


def gradient_back(oil_rate: np.ndarray, water_rate: np.ndarray) -> list:
    """Gradient Calculations Feed Backwards

    Completes a numerical gradient calculation between jet pumps. Adds a zero on
    the oil and water arrays that are passed to it. These ensures points exists for
    doing a reverse gradient calculation.

    Args:
        qoil_filt (list): Filtered Oil Prod. Rate, BOPD
        qwat_filt (list): Filtered Total Water Rate, BWPD

    Returns:
        gradient (list): Gradient of Oil-Rate / Water-Rate, bbl/bbl
    """
    oil_rate = np.append(0, oil_rate)  # add a zero to the start
    water_rate = np.append(0, water_rate)  # add a zero to the start

    if len(oil_rate) != len(water_rate):
        raise ValueError("Oil and Water Arrays Must be the same length")

    grad = []
    for i in range(len(oil_rate)):
        if i != 0:  # skip the first value of 0,0, since you aren't returning that value
            grad.append((oil_rate[i] - oil_rate[i - 1]) / (water_rate[i] - water_rate[i - 1]))
        else:
            pass
    return grad


def batch_plot_data(
    qoil_std: pd.Series,
    qwat_bpd: pd.Series,
    water: str,
    nozzles: pd.Series,
    throats: pd.Series,
    semi: pd.Series,
    wellname: str,
    coeff: tuple[float, float, float] | None,
    ax: plt.Axes,  # type: ignore
) -> None:
    """Batch Plot Data

    Create a plot to view the results from the batch run.
    Add an additional argument that would be the coefficients for
    the curve fit of the upper portion of the data.

    Args:
        qoil_std (pd.Series): Oil Prod. Rate, BOPD
        qwat_bpd (pd.Series): Water Rate, "Lift" or "Total", BPD
        water (str): Water Type, "Lift" or "Total"
        nozzles (pd.Series): Nozzle Numbers
        throats (pd.Series): Throat Ratios
        semi (pd.Series): Pandas Series of if the jet pump is a semi finalist or not
        wellname (str): Wellname String
        coeff (tuple): Exponential Curve Fit Coefficients, A, B, C
        ax (plt.Axes): Matplotlib Axes
    """
    jp_names = [noz + thr for noz, thr in zip(nozzles, throats)]  # create a list of all the jetpump names

    # plot semi-finalist
    ax.plot(qwat_bpd[semi], qoil_std[semi], marker="o", linestyle="", color="r", label="Semi-Final")

    # plot non-semi finalist
    ax.plot(qwat_bpd[~semi], qoil_std[~semi], marker="o", linestyle="", color="b", label="Eliminate")

    for qoil, qwat, jp in zip(qoil_std, qwat_bpd, jp_names):
        if not pd.isna(qoil):
            ax.annotate(jp, xy=(qwat, qoil), xycoords="data", xytext=(1.5, 1.5), textcoords="offset points")

    # show the curve fit data
    if coeff is not None:
        a, b, c = coeff  # parse out the coefficients for easier understanding
        fit_water = np.linspace(0, qwat_bpd.max(), 1000)
        fit_oil = [cf.exp_model(wat, a, b, c) for wat in fit_water]
        ax.plot(fit_water, fit_oil, color="red", linestyle="--", label="Exp. Fit")

    ax.set_xlabel(f"{water.capitalize()} Water Rate, BWPD")
    ax.set_ylabel("Produced Oil Rate, BOPD")
    # ax.title.set_text(f"{wellname} Jet Pump Performance")
    ax.legend()

    return None


def batch_plot_derv_base(
    marg_filt: pd.Series,
    qwat_filt: pd.Series,
    nozz_filt: pd.Series,
    thrt_filt: pd.Series,
    wellname: str,
    coeff: tuple[float, float, float],
    ax: Axes,
    mcolor: str | mcolors.Colormap,
    network: bool = False,
) -> None:
    """Batch Plot Derivative Base

    Create a plot showing the derivative / marginal oil results.
    Visualize how good the curve fit actually is for the data.
    Data needs to be filtered prior to passing to this function.

    Args:
        marg_filt (pd.Series): Filtered Marginal Oil - Water Ratio Rate, BOWPD
        qwat_filt (pd.Series): Filtered Water Rate, "Lift" or "Total", BPD
        nozz_filt (pd.Series): Filtered Nozzle Numbers
        thrt_filt (pd.Series): Filtered Throat Ratios
        wellname (str): Wellname String
        coeff (tuple): Exponential Curve Fit Coefficients, A, B, C
        ax (plt.Axes): Matplotlib Axes
        mcolor (str): Matplotlib Color
        network (bool): Will this be used in a network plot or not
    """
    if network:
        label1 = wellname
        label2 = "_hidden"
    else:
        label1 = "Numerical"
        label2 = "Analytical"

    ax.plot(qwat_filt, marg_filt, marker="o", linestyle="", color=mcolor, label=label1)

    jp_filt = [noz + thr for noz, thr in zip(nozz_filt, thrt_filt)]  # create a list of all the jetpump names
    for marg, qwat, jp in zip(marg_filt, qwat_filt, jp_filt):
        ax.annotate(jp, xy=(qwat, marg), xycoords="data", xytext=(1.5, 1.5), textcoords="offset points")

    a, b, c = coeff  # parse out the coefficients for easier understanding
    fit_water = np.linspace(0, qwat_filt.max(), 1000)
    fit_grad = [cf.exp_deriv(wat, b, c) for wat in fit_water]
    ax.plot(fit_water, fit_grad, color=mcolor, linestyle="--", label=label2)

    return None
