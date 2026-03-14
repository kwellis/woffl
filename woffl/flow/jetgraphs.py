"""Used for creating graphs that are dependent on multiple parameters. The functions in jet figures
ultimately call methods from jetplot to create multiple plots that are all related to each other.
"""

import os

import matplotlib.pyplot as plt
import numpy as np

from woffl.flow import jetflow as jf
from woffl.flow import jetplot as jplt
from woffl.flow import outflow as of
from woffl.flow import singlephase as sp
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.resmix import ResMix


# if you wanted to make this independent, specify pressure of the nozzle inlet in psig
# also then specify what the diffuser outlet diameter is
def choked_figures(
    tsu: float,
    rho_pf: float,
    ppf_surf: float,
    jpump_well: JetPump,
    wellbore: Pipe,
    wellprof: WellProfile,
    ipr_well: InFlow,
    prop_well: ResMix,
    folder_path: str | os.PathLike | None = None,
    rev_id: str = "rev0",
) -> None:
    """Choked Jet Pump Figures

    The following plots a jet pump with a choked throat entry. Does not correct based on
    discharge conditions or any other constraints. Creates plots of the jet pump throat entry and diffuser.

    Args:
        tsu (float): Temperature of Pump Suction, deg F
        rho_pf (float): Power Fluid Density, lbm/ft3
        ppf_surf (float): Pressure of Power Fluid at surface, psig
        jpump_well (JetPump): Jet Pump Class
        wellbore (Pipe): Pipe Class, used for diffuser diameter
        wellprof (WellProfile): WellProfile Class, for jet pump TVD
        ipr_well (InFlow): IPR Class
        prop_well (ResMix): Reservoir conditions of the well
        folder_path (Path): Path to optional output folder, saves both graphs there
        rev_id (str): A string that can be passed to keep track of revision history

    Returns:
        None
    """
    psu_min, qoil_std, te_book = jf.psu_minimize(
        tsu=tsu, ken=jpump_well.ken, ate=jpump_well.ate, ipr_su=ipr_well, prop_su=prop_well
    )

    pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)

    pte, ptm, pdi, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, prop_tm = jf.jetpump_overall(
        psu_min,
        tsu,
        pni,
        rho_pf,
        jpump_well.ken,
        jpump_well.knz,
        jpump_well.kth,
        jpump_well.kdi,
        jpump_well.ath,
        jpump_well.anz,
        wellbore.inn_area,
        ipr_well,
        prop_well,
    )

    qoil_std, te_book = jplt.throat_entry_book(psu_min, tsu, jpump_well.ken, jpump_well.ate, ipr_well, prop_well)
    vtm, di_book = jplt.diffuser_book(ptm, tsu, jpump_well.ath, jpump_well.kdi, wellbore.inn_area, qoil_std, prop_tm)

    if folder_path is not None:
        entry_name = "entry_four_" + rev_id + ".png"
        diff_name = "diffuser_four_" + rev_id + ".png"
        entry_path = os.path.join(folder_path, entry_name)
        diff_path = os.path.join(folder_path, diff_name)
    else:
        entry_path = None
        diff_path = None

    te_book.plot_te(pte_min=int(pte) - 100, fig_path=entry_path)  # don't need to see default 200
    print("Choked figures method in jetgraphs cutting before 200 psig, is this intentional?")
    di_book.plot_di(fig_path=diff_path)


def pump_pressure_relation(
    tsu: float,
    rho_pf: float,
    ppf_surf: float,
    jpump_well: JetPump,
    wellbore: Pipe,
    wellprof: WellProfile,
    ipr_well: InFlow,
    prop_well: ResMix,
    fig_path: str | os.PathLike | None = None,
) -> None:
    """Jet Pump Pressure Relationship

    Plots a jet pump and the relationship that the suction pressure has with the throat entry,
    throat exit, and diffuser pressure. Used for visualizing why modifying the suction pressure
    can be used for finding a discharge pressure that minimizes the discharge residual.

    Args:
        tsu (float): Temperature of Pump Suction, deg F
        rho_pf (float): Power Fluid Density, lbm/ft3
        ppf_surf (float): Pressure of Power Fluid at surface, psig
        jpump_well (JetPump): Jet Pump Class
        wellbore (Pipe): Pipe Class, used for diffuser diameter
        wellprof (WellProfile): WellProfile Class, for jet pump TVD
        ipr_well (InFlow): IPR Class
        prop_well (ResMix): Reservoir conditions of the well
        fig_path (Path): Path to optional output file, saves files to be viewed later

    Returns:
        None
    """
    psu_min, qoil_std, te_book = jf.psu_minimize(tsu, jpump_well.ken, jpump_well.ate, ipr_well, prop_well)
    psu_max = ipr_well.pres - 10

    psu_list = np.linspace(psu_min, psu_max, 25)

    pte_list = []
    ptm_list = []
    pdi_list = []
    qoil_list = []

    pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)  # static

    for psu in psu_list:
        pte, ptm, pdi, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, prop_tm = jf.jetpump_overall(
            psu,
            tsu,
            pni,
            rho_pf,
            jpump_well.ken,
            jpump_well.knz,
            jpump_well.kth,
            jpump_well.kdi,
            jpump_well.ath,
            jpump_well.anz,
            wellbore.inn_area,
            ipr_well,
            prop_well,
        )

        pte_list.append(pte)
        ptm_list.append(ptm)
        pdi_list.append(pdi)
        qoil_list.append(qoil_std)

    marker_style = "."
    line_style = "-"
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(psu_list, pdi_list, label="Discharge", marker=marker_style, linestyle=line_style)
    ax.plot(psu_list, ptm_list, label="Throat Exit", marker=marker_style, linestyle=line_style)
    ax.plot(psu_list, pte_list, label="Throat Entry", marker=marker_style, linestyle=line_style)
    ax.set_xlabel("Suction Pressure, psig")
    ax.set_ylabel("Pressure, psig")
    # plt.title("Comparison of Jet Pump Pressures Against Suction")
    ax.legend()
    plt.subplots_adjust(left=0.2, bottom=0.135, right=0.975, top=0.975, wspace=0.2, hspace=0.15)
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches="tight", dpi=300)
    else:
        plt.show()
    return None


def discharge_check(
    surf_pres: float,
    form_temp: float,
    rho_pf: float,
    ppf_surf: float,
    jpump_well: JetPump,
    tube: Pipe,
    wellprof: WellProfile,
    ipr_well: InFlow,
    prop_well: ResMix,
) -> None:
    """Discharge Check Choked Conditions

    The following function compares what the jet pump can discharge compared to what the
    discharge pressure needs to be to lift the well. It only looks at choked conditions of
    the jet pump instead of iterating to find a zero residual. Prints results as text, creates
    plots of the jet pump throat entry and diffuser.

    Args:
        surf_pres (float): Well Head Surface Pressure, psig
        form_temp (float): Formation Temperature, deg F
        rho_pf (float): Power Fluid Density, lbm/ft3
        ppf_surf (float): Pressure of Power Fluid at surface, psig
        jpump_well (JetPump): Jet Pump Class
        tube (Pipe): Pipe Class
        wellprof (WellProfile): Well Profile Class
        ipr_well (InFlow): IPR Class
        prop_well (ResMix): Reservoir conditions of the well

    Returns:
        None
    """
    psu_min, qoil_std, te_book = jf.psu_minimize(
        tsu=form_temp, ken=jpump_well.ken, ate=jpump_well.ate, ipr_su=ipr_well, prop_su=prop_well
    )
    pte, vte, rho_te, mach_te = te_book.dete_zero()
    pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)
    vnz = jf.nozzle_velocity(pni, pte, jpump_well.knz, rho_pf)

    qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, jpump_well.anz)
    wc_tm, qwat_su = jf.throat_wc(qoil_std, prop_well.wc, qnz_bpd)

    prop_tm = ResMix(wc_tm, prop_well.fgor, prop_well.oil, prop_well.wat, prop_well.gas)
    ptm = jf.throat_discharge(
        pte, form_temp, jpump_well.kth, vnz, jpump_well.anz, rho_pf, vte, jpump_well.ate, rho_te, prop_tm
    )
    vtm, pdi = jf.diffuser_discharge(ptm, form_temp, jpump_well.kdi, jpump_well.ath, tube.inn_area, qoil_std, prop_tm)

    md_seg, prs_ray, slh_ray = of.top_down_press(surf_pres, form_temp, qoil_std, prop_tm, tube, wellprof)

    outflow_pdi = prs_ray[-1]
    diff_pdi = pdi - outflow_pdi

    if diff_pdi >= 0:
        pdi_str = f"Well will flow choked, discharge pressure is {round(diff_pdi, 0)} psig greater than required"
    else:
        pdi_str = f"Well will NOT flow choked, discharge pressure is {round(diff_pdi, 0)} psig below the required"

    print(f"Suction Pressure: {round(psu_min, 1)} psig")
    print(f"Oil Flow: {round(qoil_std, 1)} bopd")
    print(f"Nozzle Inlet Pressure: {round(pni, 1)} psig")
    print(f"Throat Entry Pressure: {round(pte, 1)} psig")
    print(f"Throat Discharge Pressure: {round(ptm, 1)} psig")
    print(f"Required Diffuser Discharge Pressure: {round(prs_ray[-1], 1)} psig")
    print(f"Supplied Diffuser Discharge Pressure: {round(pdi, 1)} psig")
    print(pdi_str)
    print(f"Power Fluid Rate: {round(qnz_bpd, 1)} bwpd")
    print(f"Nozzle Velocity: {round(vnz, 1)} ft/s")
    print(f"Throat Entry Velocity: {round(vte, 1)} ft/s")

    # add the outflow, with the liquid holdup and pressure

    # graphing some outputs for visualization
    qsu_std, te_book = jplt.throat_entry_book(psu_min, form_temp, jpump_well.ken, jpump_well.ate, ipr_well, prop_well)
    te_book.plot_te()
    # print(te_book)
    vtm, di_book = jplt.diffuser_book(ptm, form_temp, jpump_well.ath, jpump_well.kdi, tube.inn_area, qsu_std, prop_tm)
    di_book.plot_di()
    # print(di_book)
    # te_book.plot()
