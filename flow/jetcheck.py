import numpy as np

from flow import jetflow as jf
from flow import jetplot as jplt
from flow import outflow as of
from flow import singlephase as sp
from flow.inflow import InFlow
from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe
from geometry.wellprofile import WellProfile
from pvt.resmix import ResMix


# writing a function that can be ran to easily compare current jet pump
# performance to expected performance
def jet_check(
    form_temp: float,
    jpump_tvd: float,
    rho_pf: float,
    ppf_surf: float,
    jpump_well: JetPump,
    tube: Pipe,
    ipr_well: InFlow,
    prop_well: ResMix,
) -> None:
    psu_min, qsu_std, pte, rho_te, vte = jf.psu_minimize(
        tsu=form_temp, ken=jpump_well.ken, ate=jpump_well.ate, ipr_su=ipr_well, prop_su=prop_well
    )
    # pni = jf.pf_press_depth(rho_pf, ppf_surf, jpump_tvd)
    pni = ppf_surf + sp.diff_press_static(rho_pf, jpump_tvd)
    vnz = jf.nozzle_velocity(pni, pte, jpump_well.knz, rho_pf)

    qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, jpump_well.anz)
    wc_tm = jf.throat_wc(qsu_std, prop_well.wc, qnz_bpd)

    prop_tm = ResMix(wc_tm, prop_well.fgor, prop_well.oil, prop_well.wat, prop_well.gas)
    ptm = jf.throat_discharge(
        pte, form_temp, jpump_well.kth, vnz, jpump_well.anz, rho_pf, vte, jpump_well.ate, rho_te, prop_tm
    )
    vtm, pdi = jf.diffuser_discharge(ptm, form_temp, jpump_well.kdi, jpump_well.ath, tube.inn_area, qsu_std, prop_tm)

    print(f"Suction Pressure: {round(psu_min, 1)} psig")
    print(f"Oil Flow: {round(qsu_std, 1)} bopd")
    print(f"Nozzle Inlet Pressure: {round(pni, 1)} psig")
    print(f"Throat Entry Pressure: {round(pte, 1)} psig")
    print(f"Throat Discharge Pressure: {round(ptm, 1)} psig")
    print(f"Diffuser Discharge Pressure: {round(pdi, 1)} psig")
    print(f"Power Fluid Rate: {round(qnz_bpd, 1)} bwpd")
    print(f"Nozzle Velocity: {round(vnz, 1)} ft/s")
    print(f"Throat Entry Velocity: {round(vte, 1)} ft/s")

    # graphing some outputs for visualization
    qsu_std, te_book = jplt.throat_entry_book(psu_min, form_temp, jpump_well.ken, jpump_well.ate, ipr_well, prop_well)
    te_book.plot_te()

    vtm, di_book = jplt.diffuser_book(ptm, form_temp, jpump_well.ath, jpump_well.kdi, tube.inn_area, qsu_std, prop_tm)
    di_book.plot_di()


# writing a function that can be ran to easily compare current jet pump
# performance to expected performance
def jet_check_two(
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
    """Jet Check Two

    Brings in the outflow node of the jet pump system, looking at the required discharge pressure
    at the wells flowrate vs what the pump can actually supply.
    """
    psu_min, qsu_std, pte, rho_te, vte = jf.psu_minimize(
        tsu=form_temp, ken=jpump_well.ken, ate=jpump_well.ate, ipr_su=ipr_well, prop_su=prop_well
    )
    pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)
    vnz = jf.nozzle_velocity(pni, pte, jpump_well.knz, rho_pf)

    qnz_ft3s, qnz_bpd = jf.nozzle_rate(vnz, jpump_well.anz)
    wc_tm = jf.throat_wc(qsu_std, prop_well.wc, qnz_bpd)

    prop_tm = ResMix(wc_tm, prop_well.fgor, prop_well.oil, prop_well.wat, prop_well.gas)
    ptm = jf.throat_discharge(
        pte, form_temp, jpump_well.kth, vnz, jpump_well.anz, rho_pf, vte, jpump_well.ate, rho_te, prop_tm
    )
    vtm, pdi = jf.diffuser_discharge(ptm, form_temp, jpump_well.kdi, jpump_well.ath, tube.inn_area, qsu_std, prop_tm)

    md_seg, prs_ray, slh_ray = of.top_down_press(surf_pres, form_temp, qsu_std, prop_tm, tube, wellprof)

    outflow_pdi = prs_ray[-1]
    diff_pdi = pdi - outflow_pdi

    if diff_pdi >= 0:
        pdi_str = f"Well will flow, discharge pressure is {round(diff_pdi, 0)} psig greater than required"
    else:
        pdi_str = f"Well will NOT flow, discharge pressure is {round(diff_pdi, 0)} psig below the required"

    print(f"Suction Pressure: {round(psu_min, 1)} psig")
    print(f"Oil Flow: {round(qsu_std, 1)} bopd")
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


def jetpump_solver(
    pwh: float,
    tsu: float,
    rho_pf: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: Pipe,
    wellprof: WellProfile,
    ipr: InFlow,
    prop: ResMix,
) -> None:
    """JetPump Solver

    Find a solution for the jetpump system that factors in the wellhead pressure and reservoir conditions.
    The solver will move along the psu and dEte curves until a solution is found that satisfies the outflow
    tubing and pump conditions.

    Args:
        pwh (float): Pressure Wellhead, psig
        tsu (float): Temperature Suction, deg F
        rho_pf (float): Density of the power fluid, lbm/ft3
        ppf_surf (float): Pressure Power Fluid Surface, psig
        jpump (JetPump): Jet Pump Class
        wellbore (Pipe): Pipe Class of the Wellbore
        wellprof (WellProfile): Well Profile Class
        ipr (InFlow): Inflow Performance Class
        prop (ResMix): Reservoir Mixture Conditions

    Returns:
        psu (float): Suction Pressure, psig
        qsu_std (float): Oil Flow, bopd
        what do I want to store?

    """
    # start here, and record psu_min as the starting point
    psu_min, qsu_std, pte, rho_te, vte = jf.psu_minimize(
        tsu=tsu, ken=jpump.ken, ate=jpump.ate, ipr_su=ipr, prop_su=prop
    )

    residual = system_residual(psu_min, pwh, tsu, rho_pf, ppf_surf, jpump, wellbore, wellprof, ipr, prop)

    if residual >= 0:
        status_te = "Choked"
        status_of = "Flows"
    else:
        rawr = 1

    pass


def system_residual(
    psu: float,
    pwh: float,
    tsu: float,
    rho_pf: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: Pipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
) -> float:
    """System Residual

    Solve for the jet pump system residual, which is the difference between discharge pressure
    calculated by the jetpump and the discharge pressure from the outflow.

    Args:
        psu (float): Pressure Suction, psig
        pwh (float): Pressure Wellhead, psig
        tsu (float): Temperature Suction, deg F
        rho_pf (float): Density of the power fluid, lbm/ft3
        ppf_surf (float): Pressure Power Fluid Surface, psig
        jpump (JetPump): Jet Pump Class
        wellbore (Pipe): Pipe Class of the Wellbore
        wellprof (WellProfile): Well Profile Class
        ipr_su (InFlow): Inflow Performance Class
        prop_su (ResMix): Reservoir Mixture Conditions

    Returns:
        residual (float): Jet Pump Discharge minus Out Flow Discharge, psid
    """
    pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)  # static

    # jet pump section
    pte, ptm, pdi_jp, qoil_std, prop_tm = jf.jetpump_overall(
        psu,
        tsu,
        pni,
        rho_pf,
        jpump.ken,
        jpump.knz,
        jpump.kth,
        jpump.kdi,
        jpump.ath,
        jpump.anz,
        wellbore.inn_area,
        ipr_su,
        prop_su,
    )

    # out flow section
    md_seg, prs_ray, slh_ray = of.top_down_press(pwh, tsu, qoil_std, prop_tm, wellbore, wellprof)

    pdi_of = prs_ray[-1]  # discharge pressure outflow
    return pdi_jp - pdi_of  # what the jetpump puts out vs what is required
