"""System Operations

Code that has to do with running the entire system together. A combination of the IPR,
PVT, JetPump and Outflow. Used to create a final solution to compare.
"""

import numpy as np

from woffl.flow import InFlow
from woffl.flow import jetflow as jf
from woffl.flow import outflow as of
from woffl.flow import singlephase as sp
from woffl.geometry import JetPump, Pipe, PipeInPipe, WellProfile
from woffl.pvt.resmix import ResMix


def discharge_residual(
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
) -> tuple[float, float, float, float, float]:
    """Discharge Residual

    Solve for the jet pump discharge residual, which is the difference between discharge pressure
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
        res_di (float): Jet Pump Discharge minus Out Flow Discharge, psid
        qoil_std (float): Oil Rate, STBOPD
        fwat_bpd (float): Formation Water Rate, BWPD
        qnz_bpd (float): Power Fluid Rate, BWPD
        mach_te (float): Throat Entry Mach, unitless
    """
    # also pump out the mach value at the throat entry?
    pni = ppf_surf + sp.diff_press_static(rho_pf, wellprof.jetpump_vd)  # static

    # jet pump section
    pte, ptm, pdi_jp, qoil_std, fwat_bwpd, qnz_bwpd, mach_te, prop_tm = jf.jetpump_overall(
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
    res_di = pdi_jp - pdi_of  # what the jetpump puts out vs what is required
    return res_di, qoil_std, fwat_bwpd, qnz_bwpd, mach_te


def jetpump_solver(
    pwh: float,
    tsu: float,
    rho_pf: float,  # remove this, will just use reservoir mixture water instead
    ppf_surf: float,
    jpump: JetPump,
    wellbore: Pipe,  # modify this to be annulus instead of wellbore
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
) -> tuple[float, bool, float, float, float, float]:
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
        ipr_su (InFlow): Inflow Performance Class
        prop_su (ResMix): Reservoir Mixture Conditions

    Returns:
        psu (float): Suction Pressure, psig
        sonic_status (boolean): Is the throat entry at sonic velocity?
        qoil_std (float): Oil Rate, STBOPD
        fwat_bwpd (float): Formation Water Rate, BWPD
        qnz_bwpd (float): Power Fluid Rate, BWPD
        mach_te (float): Throat Entry Mach, unitless
    """
    psu_min, qoil_std, te_book = jf.psu_minimize(tsu=tsu, ken=jpump.ken, ate=jpump.ate, ipr_su=ipr_su, prop_su=prop_su)
    res_min, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
        psu_min, pwh, tsu, rho_pf, ppf_surf, jpump, wellbore, wellprof, ipr_su, prop_su
    )

    # if the jetpump (available) discharge is above the outflow (required) discharge at lowest suction
    # the well will flow, but at its critical limit
    if res_min > 0:
        sonic_status = True
        return psu_min, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te

    psu_max = ipr_su.pres - 10  # max suction pressure that can be used
    res_max, *etc = discharge_residual(psu_max, pwh, tsu, rho_pf, ppf_surf, jpump, wellbore, wellprof, ipr_su, prop_su)

    # if the jetpump (available) discharge is below the outflow (required) discharge at highest suction
    # the well will not flow, need to pick different parameters
    if res_max < 0:
        # this isn't actually a value error, the code is working as intended
        # this provides a quick fix in the try statement in batch run
        raise ValueError("well cannot lift at max suction pressure")
        return np.nan, False, np.nan, np.nan, np.nan, np.nan

    # start secant hunting for the answer, in between the two points
    psu_list = [psu_min, psu_max]
    res_list = [res_min, res_max]

    psu_diff = 5  # criteria for when you've converged to an answer
    n = 0  # loop counter

    while abs(psu_list[-2] - psu_list[-1]) > psu_diff:
        psu_nxt = jf.psu_secant(psu_list[0], psu_list[1], res_list[0], res_list[1])
        res_nxt, qoil_std, fwat_bwpd, qnz_bwpd, mach_te = discharge_residual(
            psu_nxt, pwh, tsu, rho_pf, ppf_surf, jpump, wellbore, wellprof, ipr_su, prop_su
        )
        psu_list.append(psu_nxt)
        res_list.append(res_nxt)
        n += 1
        if n == 10:
            raise ValueError("Suction Pressure for Overall System did not converge")
    return psu_list[-1], False, qoil_std, fwat_bwpd, qnz_bwpd, mach_te
