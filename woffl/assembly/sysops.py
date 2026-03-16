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
from woffl.pvt import FormWater, ResMix


def powerfluid_residual(
    qpf_guess: float,
    pte: float,
    ppf_surf: float,
    tpf_surf: float,
    dp_stat: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    prop_pf: FormWater,
    flowpath: str,
) -> tuple[float, float, float]:
    """Power Fluid Residual

    Solve for the power fluid residual, which is the difference between the amount of power fluid
    it was guessed the jet pump needs and the amount actually delivered. This will be thrown in a
    secant loop to drive the power fluid residual to zero

    Args:
        qpf_guess (float): Power Fluid Flowrate Guess, BWPD
        pte (float): Throat Entry Pressure, psig
        ppf_surf (float): Pressure Power Fluid Surface, psig
        tpf_surf (float): Temp Power Fluid Surface, deg F
        dp_stat (float): Static Pressure, psi
        jpump (JetPump): Jet Pump Class
        wellbore (PipeInPipe): Wellbore Geometry of Tubing and Casing
        wellprof (WellProfile): Well Profile Class
        prop_pf (FormWater): Power Fluid Properties, assumed to be the same as formation water
        flowpath (str): Where the flow is occuring, either "tubing" or "annulus"

    Returns:
        qpf_residual (float): Power Fluid Residual, Guess - Calc, BWPD
        vnz (float): Nozzle Velocity, ft/s
        pni (float): Nozzle Inlet Pressure, psig
    """
    dp_fric = of.powerfluid_top_down_friction(ppf_surf, tpf_surf, qpf_guess, prop_pf, wellbore, wellprof, flowpath)
    pni = ppf_surf - dp_stat - dp_fric
    vnz = jf.nozzle_velocity(pni, pte, jpump.knz, prop_pf.density)
    _, qpf_calc = jf.nozzle_rate(vnz, jpump.anz)  # bwpd, power fluid flowrate
    return qpf_guess - qpf_calc, vnz, pni


def qpf_secant(qpf1: float, qpf2: float, res1: float, res2: float) -> float:
    """Power Fluid Secant Method

    Uses the secant method to calculate the next powerfluid to find where the pressure drop
    across the wellbore and jetpump nozzle are the same. This probably could be simplified using
    Newton method, but finding the derivative of the Serghide equation seemed daunting...

    Args:
        qpf1 (float): Power Fluid Flow One, BWPD
        qpf2 (float): Power Fluid Flow Two, BWPD
        res1 (float): Residual Power Fluid One, BWPD
        res2 (float): Residual Power Fluid Two, BWPD

    Return:
        qpf3 (float): Power Fluid Flow Three, BWPD
    """
    qpf3 = qpf2 - res2 * (qpf1 - qpf2) / (res1 - res2)
    return qpf3


def discharge_residual(
    psu: float,
    pwh: float,
    tsu: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
    prop_pf: FormWater,
    jpump_direction: str = "reverse",
) -> tuple[float, float, float, float, float]:
    """Discharge Residual

    Solve for the jet pump discharge residual, which is the difference between discharge pressure
    calculated by the jetpump and the discharge pressure from the outflow. Also iterates on the
    power fluid rate to account for pressure drop as the power fluid goes to the jet pump.

    Args:
        psu (float): Pressure Suction, psig
        pwh (float): Pressure Wellhead, psig
        tsu (float): Temperature Suction, deg F
        ppf_surf (float): Pressure Power Fluid Surface, psig
        jpump (JetPump): Jet Pump Class
        wellbore (PipeInPipe): Wellbore Geometry of Tubing and Casing
        wellprof (WellProfile): Well Profile Class
        ipr_su (InFlow): Inflow Performance Class
        prop_su (ResMix): Reservoir Mixture Conditions
        prop_pf (FormWater): Power Fluid Properties, assumed to be the same as formation water
        jpump_direction (str): Jet Pump Direction, "forward" or "reverse" Circulating

    Returns:
        res_di (float): Jet Pump Discharge minus Out Flow Discharge, psid
        qoil_std (float): Oil Rate, STBOPD
        fwat_bpd (float): Formation Water Rate, BWPD
        qnz_bpd (float): Power Fluid Rate, BWPD
        mach_te (float): Throat Entry Mach, unitless
    """
    direction_list = ["forward", "reverse"]
    if jpump_direction not in direction_list:
        raise ValueError(f"{jpump_direction} not recognized, select from {direction_list}")

    if jpump_direction == "reverse":
        production_flowpath = "tubing"
        powerfluid_flowpath = "annulus"
    else:
        production_flowpath = "annulus"
        powerfluid_flowpath = "tubing"

    # merge from jetpump_base_calcs into here to allow for power fluid flow iteration
    qoil_std, te_book = jf.throat_entry_zero_tde(
        psu=psu, tsu=tsu, ken=jpump.ken, ate=jpump.ate, ipr_su=ipr_su, prop_su=prop_su
    )
    pte, vte, rho_te, mach_te = te_book.dete_zero()

    dp_stat = sp.diff_press_static(prop_pf.density, -1 * wellprof.jetpump_vd)  # static power fluid pressure
    qpf_list = [1000.0, 2000.0]  # bwpd, power fluid flowrate guess at 1 and 2
    res_list = []  # power fluid residual list

    for qpf in qpf_list:
        qpf_res, vnz, pni = powerfluid_residual(
            qpf, pte, ppf_surf, tsu, dp_stat, jpump, wellbore, wellprof, prop_pf, powerfluid_flowpath
        )
        res_list.append(qpf_res)

    while abs(res_list[-1]) > 5:
        qpf = qpf_secant(qpf_list[-2], qpf_list[-1], res_list[-2], res_list[-1])
        qpf_res, vnz, pni = powerfluid_residual(
            qpf, pte, ppf_surf, tsu, dp_stat, jpump, wellbore, wellprof, prop_pf, powerfluid_flowpath
        )
        qpf_list.append(qpf)
        res_list.append(qpf_res)

    # should I do something with pni???
    qnz_bwpd = qpf_list[-1]
    wc_tm, fwat_bwpd = jf.throat_wc(qoil_std, prop_su.wc, qnz_bwpd)

    prop_tm = ResMix(wc_tm, prop_su.fgor, prop_su.oil, prop_su.wat, prop_su.gas)
    ptm = jf.throat_discharge(pte, tsu, jpump.kth, vnz, jpump.anz, prop_pf.density, vte, jpump.ate, rho_te, prop_tm)
    # diffuser area is assumed to be the same as the tubing area, whether forward or reverse jet pump
    vtm, pdi_jp = jf.diffuser_discharge(ptm, tsu, jpump.kdi, jpump.ath, wellbore.inn_pipe.inn_area, qoil_std, prop_tm)

    # out flow section
    md_seg, prs_ray, slh_ray = of.production_top_down_press(
        pwh, tsu, qoil_std, prop_tm, wellbore, wellprof, production_flowpath
    )

    pdi_of = prs_ray[-1]  # discharge pressure outflow
    res_di = pdi_jp - pdi_of  # what the jetpump puts out vs what is required
    return res_di, qoil_std, fwat_bwpd, qnz_bwpd, mach_te


def jetpump_solver(
    pwh: float,
    tsu: float,
    ppf_surf: float,
    jpump: JetPump,
    wellbore: PipeInPipe,
    wellprof: WellProfile,
    ipr_su: InFlow,
    prop_su: ResMix,
    prop_pf: FormWater,
    jpump_direction: str = "reverse",
) -> tuple[float, bool, float, float, float, float]:
    """JetPump Solver

    Find a solution for the jetpump system. Takes into account the reservoir conditions, the wellhead
    pressure, tubing and annulus geometry. The solver will move along the psu and dEte curves until a
    solution is found that satisfies the outflow tubing, annulus and pump conditions.

    Args:
        pwh (float): Pressure Wellhead, psig
        tsu (float): Temperature Suction, deg F
        ppf_surf (float): Pressure Power Fluid Surface, psig
        jpump (JetPump): Jet Pump Class
        wellbore (PipeInPipe): Wellbore Geometry of Tubing and Casing
        wellprof (WellProfile): Well Profile Class
        ipr_su (InFlow): Inflow Performance Class
        prop_su (ResMix): Reservoir Mixture Conditions
        prop_pf (FormWater): Power Fluid Properties, assumed to be the same as formation water
        jpump_direction (str): Jet Pump Direction, "forward" or "reverse" Circulating

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
        psu_min, pwh, tsu, ppf_surf, jpump, wellbore, wellprof, ipr_su, prop_su, prop_pf, jpump_direction
    )

    # if the jetpump (available) discharge is above the outflow (required) discharge at lowest suction
    # the well will flow, but at its critical limit
    if res_min > 0:
        sonic_status = True
        return psu_min, sonic_status, qoil_std, fwat_bwpd, qnz_bwpd, mach_te

    psu_max = ipr_su.pres - 10  # max suction pressure that can be used
    res_max, *etc = discharge_residual(
        psu_max, pwh, tsu, ppf_surf, jpump, wellbore, wellprof, ipr_su, prop_su, prop_pf, jpump_direction
    )

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
            psu_nxt, pwh, tsu, ppf_surf, jpump, wellbore, wellprof, ipr_su, prop_su, prop_pf, jpump_direction
        )
        psu_list.append(psu_nxt)
        res_list.append(res_nxt)
        n += 1
        if n == 10:
            raise ValueError("Suction Pressure for Overall System did not converge")
    return psu_list[-1], False, qoil_std, fwat_bwpd, qnz_bwpd, mach_te
