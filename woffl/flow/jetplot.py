"""Jet Plotting and Storage

A series of functions that are predominately built for variable storage and plotting.
Classes that store values and results are referred to as Books. A single book class is
built that stores calculations for the throat entry book and a diffuser. The books possess
the following.

Returns:
        prs_ray (np array): Pressure Array, psig
        vel_ray (np array): Velocity Array, ft/s
        rho_ray (np array): Density Array, lbm/ft3
        snd_ray (np array): Speed of Sound Array, ft/s
        mach_ray (np array): Mach Number, unitless
        kde_ray (np array): Kinetic Differential Energy, ft2/s2
        ede_ray (np array): Expansion Differential Energy, ft2/s2
        tde_ray (np array): Total Differntial Energy, ft2/s2
        grad_ray (np array): Gradient of tde/dp Array, ft2/(s2*psig)
"""

import os
from typing import Any, cast

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from woffl.flow import jetflow as jf  # legacy
from woffl.flow import singlephase as sp
from woffl.flow.inflow import InFlow
from woffl.pvt.resmix import ResMix


class JetBook:
    def __init__(self, prs: float, vel: float, rho: float, snd: float, kde: float):
        """Book for storing Jet Pump Enterance / Diffuser Calculations

        Create a book for storing results values from the throat enterance and diffuser.
        Can be used later for graphing and other analysis.

        Args:
            prs (float): Pressure, psig
            vel (float): Velocity, ft/s
            rho (float): Density, lbm/ft3
            snd (float): Speed of Sound, ft/s
            kde (float): Kinetic Differential Energy, ft2/s2
        """
        self.prs_ray = np.array([prs])
        self.vel_ray = np.array([vel])
        self.rho_ray = np.array([rho])
        self.snd_ray = np.array([snd])

        self.kde_ray = np.array([kde])

        ede = 0
        tde = kde + ede

        self.ede_ray = np.array([ede])  # expansion energy array
        self.tde_ray = np.array([tde])  # total differential energy
        self.mach_ray = np.array([vel / snd])  # mach number
        self.grad_ray = np.array([np.nan])  # gradient of tde vs prs

    # https://docs.python.org/3/library/string.html#formatspec
    def __repr__(self):
        """Creates a fancy table to see some of the stored data"""
        sformat = "{:>8} | {:>8} | {:>7} | {:>6} | {:>6} \n"  # string format
        nformat = "{:>8.1f} | {:>8.1f} | {:>7.1f} | {:>6.1f} | {:>6.2f} \n"  # number format
        spc = 48 * "-" + "\n"  # spacing
        pout = sformat.format("pressure", "velocity", "density", "sound", "mach")
        pout = pout + sformat.format("psig", "ft/s", "lbm/ft3", "ft/s", "no_u") + spc
        for prs, vel, rho, snd, mach in zip(self.prs_ray, self.vel_ray, self.rho_ray, self.snd_ray, self.mach_ray):
            pout += nformat.format(prs, vel, rho, snd, mach)
        return pout

    def append(self, prs: float, vel: float, rho: float, snd: float, kde: float):
        """Append Values onto the Throat Entry Book

        Appended values are added onto the existing throat entry arrays.

        Args:
            prs (float): Pressure, psig
            vel (float): Velocity, ft/s
            rho (float): Density, lbm/ft3
            snd (float): Speed of Sound, ft/s
            kde (float): Kinetic Differential Energy, ft2/s2
        """
        self.prs_ray = np.append(self.prs_ray, prs)
        self.vel_ray = np.append(self.vel_ray, vel)
        self.rho_ray = np.append(self.rho_ray, rho)
        self.snd_ray = np.append(self.snd_ray, snd)
        self.kde_ray = np.append(self.kde_ray, kde)

        ede = self.ede_ray[-1] + jf.incremental_ee(self.prs_ray[-2:], self.rho_ray[-2:])
        tde = kde + ede

        self.ede_ray = np.append(self.ede_ray, ede)
        self.tde_ray = np.append(self.tde_ray, tde)
        self.mach_ray = np.append(self.mach_ray, vel / snd)  # mach number

        grad = (self.tde_ray[-2] - self.tde_ray[-1]) / (self.prs_ray[-2] - self.prs_ray[-1])
        self.grad_ray = np.append(self.grad_ray, grad)  # gradient of tde vs prs

    def plot_te(self, pte_min=200, fig_path: str | os.PathLike | None = None) -> None:
        """Throat Entry Plots

        Create a series of graphs to use for visualization of the results
        in the book from the throat entry area.

        Args:
            pte_min (float): Minimum Throat Entrance Pressure to Display, psig
            fig_path (Path): Path to optional output file, saves files to be viewed later
        """
        mask = self.prs_ray >= pte_min

        fig, axs = self._throat_entry_graphs(
            self.prs_ray[mask],
            self.vel_ray[mask],
            self.rho_ray[mask],
            self.snd_ray[mask],
            self.kde_ray[mask],
            self.ede_ray[mask],
            self.tde_ray[mask],
            self.grad_ray[mask],
        )
        plt.tight_layout()  # Apply tight layout

        if fig_path is not None:
            plt.savefig(fig_path, bbox_inches="tight", dpi=300)
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()
        return None

    def plot_di(self, fig_path: str | os.PathLike | None = None) -> None:
        """Diffuser Plots

        Create a series of graphs to use for visualization of the results in
        the book from the diffuser area.

        Args:
            fig_path (Path): Path to optional output file, saves files to be viewed later
        """
        fig, axs = self._diffuser_graphs(
            self.prs_ray,
            self.vel_ray,
            self.rho_ray,
            self.snd_ray,
            self.kde_ray,
            self.ede_ray,
            self.tde_ray,
        )
        plt.tight_layout()  # Apply tight layout

        if fig_path is not None:
            plt.savefig(fig_path, bbox_inches="tight", dpi=300)
            plt.close(fig)  # Close the figure to free memory
        else:
            plt.show()
        return None

    def dete_zero(self) -> tuple[float, float, float, float]:
        """Throat Entry Parameters at Zero Total Differential Energy

        Args:
            None

        Return:
            pte (float): Throat Entry Pressure, psig
            vte (float): Throat Entry Velocity, ft/s
            rho_te (float): Throat Entry Density, lbm/ft3
            mach_te (float): Mach Throat Entry, unitless
        """
        return self._dete_zero(self.prs_ray, self.vel_ray, self.rho_ray, self.tde_ray, self.mach_ray)

    def dedi_zero(self) -> float:
        """Diffuser Discharge Pressure at Zero Total Differential Energy

        Args:
            None

        Return:
            pdi (float): Diffuser Discharge Pressure, psig
        """
        return np.interp(0, self.tde_ray, self.prs_ray)  # type: ignore

    @staticmethod
    def _dete_zero(
        prs_ray: np.ndarray,
        vel_ray: np.ndarray,
        rho_ray: np.ndarray,
        tde_ray: np.ndarray,
        mach_ray: np.ndarray,
    ) -> tuple[float, float, float, float]:
        """Throat Entry Parameters at Zero Total Differential Energy

        Calculate the throat entry pressure, density, and velocity where dEte is zero

        Args:
            prs_ray (np array): Pressure Array, psig
            vel_ray (np array): Velocity Array, ft/s
            rho_ray (np array): Density Array, lbm/ft3
            tde_ray (np array): Total Differential Energy, ft2/s2

        Return:
            pte (float): Throat Entry Pressure, psig
            vte (float): Throat Entry Velocity, ft/s
            rho_te (float): Throat Entry Density, lbm/ft3
            mach_te (float): Mach Throat Entry, unitless
        """
        dtdp = np.gradient(tde_ray, prs_ray)  # uses central limit thm, so same size
        mask = dtdp >= 0  # only points where slope is greater than or equal to zero

        tde_flipped = np.flip(tde_ray[mask])

        pte = np.interp(0, tde_flipped, np.flip(prs_ray[mask]))
        vte = np.interp(0, tde_flipped, np.flip(vel_ray[mask]))
        rho_te = np.interp(0, tde_flipped, np.flip(rho_ray[mask]))
        mach_te = np.interp(0, tde_flipped, np.flip(mach_ray[mask]))

        return pte, vte, rho_te, mach_te  # type: ignore

    @staticmethod
    def _throat_entry_graphs(
        prs_ray: np.ndarray,
        vel_ray: np.ndarray,
        rho_ray: np.ndarray,
        snd_ray: np.ndarray,
        kde_ray: np.ndarray,
        ede_ray: np.ndarray,
        tde_ray: np.ndarray,
        grad_ray: np.ndarray,
    ) -> tuple[Figure, np.ndarray]:
        """
        Creates a 4-panel visualization of fluid dynamics parameters at throat entry.

        Returns:
            fig: (plt.Figure) Matplotlib figure
            axs: (plt.Axes) Matplotlib axis
        """

        mach_ray = vel_ray / snd_ray
        colors = mpl.colormaps["tab10"].colors  # type: ignore
        # grad_ray = np.gradient(tde_ray, prs_ray)
        # psu = prs_ray[0]
        idx_sort = np.argsort(mach_ray)
        pidx = np.searchsorted(mach_ray, 1, side="left", sorter=idx_sort)  # find index location where mach is 1
        pmo = float(np.interp(1, mach_ray, prs_ray))  # interpolate for pressure at mach 1, pmo
        pgo = float(np.interp(0, np.flip(grad_ray), np.flip(prs_ray)))  # find point where gradient is zero
        pte, vte, rho_te, mach_te = JetBook._dete_zero(prs_ray, vel_ray, rho_ray, tde_ray, mach_ray)

        fig, axs = plt.subplots(4, sharex=True, figsize=(5.5, 7.5))
        axs = cast(np.ndarray, axs)

        plt.rcParams["mathtext.default"] = "regular"
        # fig.suptitle(f"Suction at {round(psu,0)} psi, Mach 1 at {round(pmo,0)} psi")

        marker_style = "."
        line_style = "-"

        # first plot
        axs[0].plot(prs_ray, 1 / rho_ray, marker=marker_style, linestyle=line_style, color=colors[0])
        axs[0].set_ylabel("Spec. Vol, $ft^{3}/lb_{m}$")

        # second plot
        axs[1].plot(
            prs_ray, vel_ray, label="Mixture Velocity", marker=marker_style, linestyle=line_style, color=colors[3]
        )
        axs[1].plot(
            prs_ray, snd_ray, label="Speed of Sound", marker=marker_style, linestyle=line_style, color=colors[0]
        )
        vel_span = vel_ray[pidx] - min(vel_ray)
        axs[1].annotate(
            text="Mach 1", xy=(pmo, vel_ray[pidx] - (1 / 8) * vel_span), rotation=90, ha="center", va="top", fontsize=10
        )
        axs[1].set_ylabel("Velocity, ft/s")
        axs[1].legend()

        # third plot
        axs[2].plot(prs_ray, kde_ray, label="Kinetic", marker=marker_style, linestyle=line_style, color=colors[3])
        axs[2].plot(prs_ray, ede_ray, label="Expansion", marker=marker_style, linestyle=line_style, color=colors[0])
        axs[2].set_ylabel("Energy, $ft^{2}/s^{2}$")
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[2].legend(loc="center left")

        # fourth plot
        axs[3].plot(prs_ray, tde_ray, marker=marker_style, linestyle=line_style, color=colors[4])

        tde_span = max(tde_ray) - tde_ray[pidx]
        axs[3].annotate(
            text="Grad 0",
            xy=(pgo, tde_ray[pidx] + 1 / 16 * tde_span),
            rotation=90,
            ha="right",
            va="bottom",
            fontsize=10,
        )
        axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[3].axhline(y=0, linestyle="--", linewidth=1, color=colors[7])
        axs[3].set_ylabel("$E_{te}$, $ft^{2}/s^{2}$")
        axs[3].set_xlabel("Throat Entry Pressure, psig")
        plt.subplots_adjust(left=0.13, bottom=0.075, right=0.975, top=0.99, wspace=0.2, hspace=0.1)
        return fig, axs

    @staticmethod
    def _diffuser_graphs(
        prs_ray: np.ndarray,
        vel_ray: np.ndarray,
        rho_ray: np.ndarray,
        snd_ray: np.ndarray,
        kde_ray: np.ndarray,
        ede_ray: np.ndarray,
        tde_ray: np.ndarray,
    ) -> tuple[Figure, np.ndarray]:
        """
        Creates a 4-panel visualization of fluid dynamics parameters at diffuser.

        Returns:
            fig: (plt.Figure) Matplotlib figure
            axs: (plt.Axes) Matplotlib axis
        """

        colors = mpl.colormaps["tab10"].colors  # type: ignore
        # ptm = prs_ray[0]

        marker_style = "."
        line_style = "-"
        fig, axs = plt.subplots(4, sharex=True, figsize=(5.5, 7.5))
        axs = cast(np.ndarray, axs)
        plt.rcParams["mathtext.default"] = "regular"

        axs[0].plot(prs_ray, 1 / rho_ray, marker=marker_style, linestyle=line_style, color=colors[0])
        axs[0].set_ylabel("Spec. Vol, $ft^{3}/lb_{m}$")

        axs[1].plot(
            prs_ray, vel_ray, label="Diffuser Outlet", marker=marker_style, linestyle=line_style, color=colors[3]
        )
        axs[1].plot(
            prs_ray, snd_ray, label="Speed of Sound", marker=marker_style, linestyle=line_style, color=colors[0]
        )
        # axs[1].scatter(ptm, vtm, label="Diffuser Inlet")
        axs[1].set_ylabel("Velocity, ft/s")
        axs[1].legend(loc="center right")

        axs[2].plot(prs_ray, kde_ray, label="Kinetic", marker=marker_style, linestyle=line_style, color=colors[3])
        axs[2].plot(prs_ray, ede_ray, label="Expansion", marker=marker_style, linestyle=line_style, color=colors[0])
        # axs[2].axhline(y=0, linestyle="--", linewidth=1, color=colors[7])
        axs[2].set_ylabel("Energy, $ft^{2}/s^{2}$")
        axs[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[2].legend()

        axs[3].plot(prs_ray, tde_ray, marker=marker_style, linestyle=line_style, color=colors[4])
        axs[3].axhline(y=0, linestyle="--", linewidth=1, color=colors[7])
        axs[3].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[3].set_ylabel("$E_{di}$, $ft^{2}/s^{2}$")
        axs[3].set_xlabel("Diffuser Outlet Pressure, psig")

        """
        if max(tde_ray) >= 0 and min(tde_ray) <= 0:  # make sure a solution exists
            pdi = np.interp(0, tde_ray, prs_ray)
            vdi = np.interp(pdi, prs_ray, vel_ray)
            ycoord = (min(vel_ray) + max(snd_ray)) / 2
            axs[1].axvline(x=pdi, color="black", linestyle="--", linewidth=1)
            axs[1].annotate(text=f"{round(vdi, 1)} ft/s", xy=(pdi, ycoord), rotation=90)  # type: ignore

            ycoord = min(tde_ray)
            axs[3].axvline(x=pdi, color="black", linestyle="--", linewidth=1)
            axs[3].annotate(text=f"{int(pdi)} psi", xy=(pdi, ycoord), rotation=90)
            fig.suptitle(f"Diffuser Inlet and Outlet at {round(ptm,0)} and {round(pdi,0)} psi")  # type: ignore
        else:
            fig.suptitle(f"Diffuser Inlet at {round(ptm,0)} psi")
        """
        plt.subplots_adjust(left=0.13, bottom=0.075, right=0.975, top=0.99, wspace=0.2, hspace=0.1)
        return fig, axs


# this goes all the way down to 200 psig
def throat_entry_book(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, JetBook]:
    """Create Throat Entry Book

    Create a Book of Throat Entry Values that can be used for visualization.
    Shows what is occuring inside the throat entry while pressure is dropped.
    Keeps all the values, even when pte velocity is greater than Mach 1.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ken (float): Enterance Friction Factor, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        qoil_std (float): Oil Rate, STBOPD
        te_book (JetBook): Book of values for inside the throat entry
    """
    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd

    prop_su = prop_su.condition(psu, tsu)
    qtot = sum(prop_su.insitu_volm_flow(qoil_std))
    vte = sp.velocity(qtot, ate)

    te_book = JetBook(psu, vte, prop_su.rho_mix(), prop_su.cmix(), jf.enterance_ke(ken, vte))

    ray_len = 50  # number of elements in the array
    pte_ray = np.linspace(200, psu, ray_len)  # throat entry pressures
    pte_ray = np.flip(pte_ray, axis=0)  # start with high pressure and go low

    for pte in pte_ray[1:]:  # start with the second value, psu is the first and is used to create array

        prop_su = prop_su.condition(pte, tsu)
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)

        te_book.append(pte, vte, prop_su.rho_mix(), prop_su.cmix(), jf.enterance_ke(ken, vte))

    return qoil_std, te_book


def diffuser_book(
    ptm: float, ttm: float, ath: float, kdi: float, adi: float, qoil_std: float, prop_tm: ResMix
) -> tuple[float, JetBook]:
    """Create a Diffuser Book

    Create a book of diffuser arrays. The arrays are used to find where the diffuser
    pressure crosses the energy equilibrium mark and find discharge pressure.

    Args:
        ptm (float): Throat Mixture Pressure, psig
        ttm (float): Throat Mixture Temp, deg F
        ath (float): Throat Area, ft2
        kdi (float): Diffuser Friction Loss, unitless
        adi (float): Diffuser Area, ft2
        qoil_std (float): Oil Rate, STD BOPD
        prop_tm (ResMix): Properties of Throat Mixture

    Returns:
        vtm (float): Throat Mixture Velocity, ft/s
        di_book (JetBook): Book of values of what is occuring inside throat entry
    """
    prop_tm = prop_tm.condition(ptm, ttm)
    qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
    vtm = sp.velocity(qtot, ath)
    vdi = sp.velocity(qtot, adi)

    di_book = JetBook(ptm, vdi, prop_tm.rho_mix(), prop_tm.cmix(), jf.diffuser_ke(kdi, vtm, vdi))

    ray_len = 30
    pdi_ray = np.linspace(ptm, ptm + 1500, ray_len)  # throat entry pressures

    for pdi in pdi_ray[1:]:  # start with 2nd value, ptm already used previously

        prop_tm = prop_tm.condition(pdi, ttm)
        qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
        vdi = sp.velocity(qtot, adi)

        di_book.append(pdi, vdi, prop_tm.rho_mix(), prop_tm.cmix(), jf.diffuser_ke(kdi, vtm, vdi))

    return vtm, di_book


def multi_throat_entry_books(
    psu_ray: list | np.ndarray, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[list, list]:
    """Multiple Throat Entry Arrays

    Calculate throat entry arrays at different suction pressures. Used to
    graph later. Similiar to Figure 5 from Robert Merrill Paper.

    Args:
        psu_ray (list): List or Array of Suction Pressures, psig
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        qoil_list (list): List of Oil Rates
        book_list (list): List of Jet Book Results
    """
    if max(psu_ray) >= ipr_su.pres:
        raise ValueError("Max suction pressure must be less than reservoir pressure")
    # 200 is arbitary and has been hard coded into the throat entry array calcs
    if min(psu_ray) <= 200:
        raise ValueError("Min suction pressure must be greater than 200 psig")

    book_list = list()  # create empty list to fill up with results
    qoil_list = list()

    for psu in psu_ray:
        qoil_std, te_book = throat_entry_book(psu, tsu, ken, ate, ipr_su, prop_su)

        qoil_list.append(qoil_std)
        book_list.append(te_book)

    return qoil_list, book_list


def te_tde_subsonic_plot(qoil_std: float, te_book: JetBook, color: str) -> tuple[Axes, float, float, float, float]:
    """Throat Entry (TE) Total Differential Energy (TDE) for Subsonic Values Plot

    Args:
        qoil_std (float): Oil Rate, BOPD
        te_book (JetBook): Throat Entry Results Book
        color (string): Matplotlib Recognized Color Description

    Return:
        ax (Axis): Matplotlib Axis to be used later
        x_ipr (float): X Value of IPR Boundary
        y_ipr (float): Y Value of IPR Boundary
        x_mach (float): X Value of Mach Boundary
        y_mach (float): Y Value of Mach Boundary

    """
    # psu = te_book.prs_ray[0]
    pgo = float(np.interp(0, np.flip(te_book.grad_ray), np.flip(te_book.prs_ray)))  # find point where gradient is zero
    # pmo = np.interp(1, te_book.mach_ray, te_book.prs_ray)

    # idx_sort = np.argsort(te_book.prs_ray)
    # find index location where mach is 1
    # pmo_idx = np.searchsorted(te_book.prs_ray, pmo, side="left", sorter=idx_sort)
    tde_pgo = np.interp(pgo, np.flip(te_book.prs_ray), np.flip(te_book.tde_ray))

    pte_ray = te_book.prs_ray[te_book.prs_ray >= pgo]
    tee_ray = te_book.tde_ray[te_book.prs_ray >= pgo]

    ax = plt.gca()
    ax.scatter(pte_ray, tee_ray, color=color, marker=".", label=f"{int(qoil_std)} bopd")
    x_ipr = te_book.prs_ray[0]
    y_ipr = te_book.tde_ray[0]
    x_mach = pgo
    y_mach = tde_pgo

    ax.scatter(pgo, tde_pgo, marker=".", color=color)  # type: ignore
    return ax, float(x_ipr), float(y_ipr), float(x_mach), float(y_mach)


def func_line(x, m, b):
    """Used for curve fitting Mach / IPR Lines"""
    return m * x + b


def inv_line(y, m, b):
    """Used to find x value of pte when dete=0"""
    return (y - b) / m


def mach_annotate(x_mach: list, y_mach: list) -> None:
    """Annotate Mach Line

    Args:
        x_mach (list): List of pte values at Mach Boundary
        y_mach (list): List of dete values at Mach Boundary

    Returns:
        Nothing
    """
    # curve fit data
    popt, pcov = opt.curve_fit(func_line, x_mach, y_mach)
    x_min, x_max = min(x_mach), inv_line(0, popt[0], popt[1])
    y_min, y_max = min(y_mach), 0
    x_coord = (x_min + x_max) / 2
    y_coord = (y_min + y_max) / 2
    x_dist = x_max - x_min
    y_dist = y_max - y_min

    offset = 5 / 64
    ax = plt.gca()
    ax.plot(x_mach, y_mach, linestyle="--", linewidth=1, color="#7f7f7f")
    # ax.plot(x_coord, y_coord, marker="x")

    rot_angle = np.rad2deg(np.arctan(popt[0]))
    ax.annotate(
        text="Sonic Limit",
        xy=(x_coord - (1 / 8) * x_dist, y_coord + offset * y_dist),
        rotation=rot_angle,
        transform_rotates_text=True,
        ha="center",
        va="center",
        fontsize=10,
    )


def inflow_annotate(x_ipr: list, y_ipr: list) -> None:
    """Annotate Mach Line

    Args:
        x_ipr (list): List of pte values at IPR Boundary
        y_ipr (list): List of dete values at IPR Boundary

    Returns:
        Nothing
    """
    # curve fit data
    popt, pcov = opt.curve_fit(func_line, x_ipr, y_ipr)
    x_min, x_max = min(x_ipr), max(x_ipr)
    y_min, y_max = min(y_ipr), max(y_ipr)
    x_coord = (x_min + x_max) / 2
    y_coord = (y_min + y_max) / 2
    x_dist = x_max - x_min
    y_dist = y_max - y_min

    offset = 5 / 64
    ax = plt.gca()
    ax.plot(x_ipr, y_ipr, linestyle="--", linewidth=1, color="#7f7f7f")
    # ax.plot(x_coord + x_dist / 16, y_coord + y_dist / 16, marker="x")

    rot_angle = np.rad2deg(np.arctan(popt[0]))
    ax.annotate(
        text="Inflow Limit",
        xy=(x_coord + offset * x_dist, y_coord + offset * y_dist),
        rotation=rot_angle,
        transform_rotates_text=True,
        ha="center",
        va="center",
        fontsize=10,
    )


def multi_suction_graphs(qoil_list: list, book_list: list, fig_path: str | os.PathLike | None = None) -> None:
    """Throat Entry Graphs for Multiple Suction Pressures

    Create a graph that shows throat entry equation solutions for multiple suction pressures. The optional
    arguement fig_path allows a path to be passed to locally save the graph for later use.

    Args:
        qoil_list (list): List of Oil Rates at different suction pressures
        book_list (list): List of throat entry books at various suction pressures
        fig_path (Path): Path to output file, saves files to be viewed later

    Returns:
        Graphs
    """

    plt.rcParams["mathtext.default"] = "regular"
    prop_cycle = plt.rcParams["axes.prop_cycle"]()  # convert to iterator
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ipr_x = []
    ipr_y = []
    mach_x = []
    mach_y = []

    for qoil_std, te_book in zip(qoil_list, book_list):
        color = next(prop_cycle)["color"]  # new color method
        ax, x_ipr, y_ipr, x_mach, y_mach = te_tde_subsonic_plot(
            qoil_std, te_book, color
        )  # need parent and children classes
        ipr_x.append(x_ipr)
        ipr_y.append(y_ipr)
        mach_x.append(x_mach)
        mach_y.append(y_mach)

    mach_annotate(mach_x, mach_y)
    inflow_annotate(ipr_x, ipr_y)
    ax.set_xlabel("Throat Entry Pressure, psig")
    ax.set_ylabel("$E_{te}$, $ft^{2}/s^{2}$")
    ax.axhline(y=0, linestyle="--", linewidth=1, color="#7f7f7f")  # grey color from tab10
    ax.legend(loc="lower right")
    plt.subplots_adjust(left=0.2, bottom=0.135, right=0.975, top=0.975, wspace=0.2, hspace=0.15)
    plt.tight_layout()

    if fig_path is not None:
        plt.savefig(fig_path)
    else:
        plt.show()
    return None
