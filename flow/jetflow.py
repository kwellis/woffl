"""Jet Flow Equations

Functions that are used for solving the fluid dynamics inside a jet pump. The actual
jet pump geometry is accomplished in a seperate module.
"""

import math

import numpy as np
from scipy.integrate import trapezoid

from flow import jetplot as jp
from flow import singlephase as sp
from flow.inflow import InFlow
from pvt.resmix import ResMix


def enterance_ke(ken: float, vte: float) -> float:
    """Throat Enterance Kinetic Energy

    Calculate the kinetic energy in the throat enterance.
    Add the energy lost due to friction to balance out with EE.

    Args:
        ken: Throat Entry Friction Factor, unitless
        vte: Velocity at Throat Entry, ft/s

    Returns:
        ke_te: Throat Enterance Kinetic Energy, ft2/s2
    """
    ke_te = (1 + ken) * (vte**2) / 2
    return ke_te


def incremental_ee(prs_ray: np.ndarray, rho_ray: np.ndarray) -> float:
    """Fluid Incremental Expansion Energy

    Calculate the incremental change in expansion energy for a fluid over a
    pressure change. Uses the trapezoid rule to sum the area under the density
    curve for a difference in pressure. Equal to Sum (dp/ρ). The length of the
    arrays must match and be equal or greater than length 2.

    Args:
        prs_ray (np.ndarray): Array of pressures, psig
        rho_ray (np.ndarray): Array of densitys, lbm/ft3

    Returns:
        ee_inc (float): Expansion Energy Incremental, ft2/s2
    """
    ee_inc = trapezoid(1 / rho_ray, 144 * 32.174 * prs_ray)
    return ee_inc


# dete and dedi, dete_machone, I do all this extra tracking when all that
# is really needed is just tee_fin,


def dete_zero(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, float, float, float]:
    """Throat Entry Differential Energy at Zero

    Find the pte, rho_te and vte that correspond to a zero value of tde.
    Use IPR to find the expected well production rate at the specific psu.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        qoil_std (float): Oil Rate, STBOPD
        pte (float): Throat Entry Pressure, psig
        rho_te (float): Throat Entry Density, lbm/ft3
        vte (float): Throat Entry Velocity, ft/s
    """
    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd

    prop_su = prop_su.condition(psu, tsu)
    qtot = sum(prop_su.insitu_volm_flow(qoil_std))
    vte = sp.velocity(qtot, ate)

    te_book = jp.ThroatEnteranceBook(psu, vte, prop_su.rho_mix(), prop_su.cmix(), enterance_ke(ken, vte))

    pdec = 50  # pressure decrease

    # runs the risk that you never cross the tde, then you need a smaller psu...?
    # this can be mitigated by finding the lowest psu and never going below that?
    # keep lowering pte until tde is under zero.
    while (te_book.tde_ray[-1] > 1) and (te_book.prs_ray[-1] > pdec):
        pte = te_book.prs_ray[-1] - pdec

        prop_su = prop_su.condition(pte, tsu)
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)

        te_book.append(pte, vte, prop_su.rho_mix(), prop_su.cmix(), enterance_ke(ken, vte))

    pte, vte, rho_te = te_book.dete_zero()

    return qoil_std, pte, rho_te, vte


def dete_grad_zero(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, float, jp.ThroatEnteranceBook]:
    """dEte Gradient Zero

    Find the numerical value of the dEte (differential energy throat entry) equation
    when the gradient (slope?) is a zero value. The physical meaning of the when the slope
    is zero is when the transistion from subsonic (mach < 1) to sonic (mach > 1) flow. A
    detailed derivation that shows the difference relationship to dEte and Ma value can
    be produced by Kaelin Ellis upon request

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ken (float): Enterance Friction Factor, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        tde_fin (float): Total Differential Energy at Mach 1, ft2/s2
    """
    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd

    prop_su = prop_su.condition(psu, tsu)
    qtot = sum(prop_su.insitu_volm_flow(qoil_std))
    vte = sp.velocity(qtot, ate)

    te_book = jp.ThroatEnteranceBook(psu, vte, prop_su.rho_mix(), prop_su.cmix(), enterance_ke(ken, vte))

    pdec = 50  # pressure decrease

    # keep mach under one, and pte above pdec, so it doesn't go negative
    while (te_book.mach_ray[-1] <= 1) and (te_book.prs_ray[-1] > pdec):

        pte = te_book.prs_ray[-1] - pdec
        prop_su = prop_su.condition(pte, tsu)
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)

        te_book.append(pte, vte, prop_su.rho_mix(), prop_su.cmix(), enterance_ke(ken, vte))

    if te_book.mach_ray[-1] >= 1:
        tde_fin = np.interp(1, te_book.mach_ray, te_book.tde_ray)  # find tee where mach = 1
    else:
        tde_fin = te_book.tde_ray[-1]

    return tde_fin, qoil_std, te_book  # type: ignore


def tee_last(
    psu: float, tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Throat Enterance Energy Last Value in Array

    Calculate the amount of energy in the throat enterance when the flow
    hits sonic velocity, mach = 1 or when the pte is low on pressure (< pdec).
    The suction pressure, psu and final enterance energy are fed into a secant
    solver that finds psu that gives a zero final energy.

    Args:
        psu (float): Suction Pressure, psig
        tsu (float): Suction Temp, deg F
        ken (float): Enterance Friction Factor, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        tee_fin (float): Final Throat Energy Equation Value, ft2/s2
        qoil_std (float): Oil flow from reservoir, stbopd
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vte_ray (np array): Velocity Throat Entry Array, ft/s
        tee_ray (np array): Throat Entry Equation Array, ft2/s2
    """
    qoil_std = ipr_su.oil_flow(psu, method="pidx")  # oil standard flow, bopd
    prop_su = prop_su.condition(psu, tsu)
    qtot = sum(prop_su.insitu_volm_flow(qoil_std))
    vte = sp.velocity(qtot, ate)

    pte_ray = np.array([psu])
    vte_ray = np.array([vte])
    rho_ray = np.array([prop_su.rho_mix()])
    mach_ray = np.array([vte / prop_su.cmix()])

    kse_ray = np.array([enterance_ke(ken, vte)])
    ese_ray = np.array([0])  # initial pe is zero
    tee_ray = np.array([kse_ray + ese_ray])

    pdec = 50  # pressure decrease

    while (
        mach_ray[-1] <= 1 and pte_ray[-1] > pdec
    ):  # keep mach under one, and pte above pdec, so it doesn't go negative
        pte = pte_ray[-1] - pdec
        prop_su = prop_su.condition(pte, tsu)
        qtot = sum(prop_su.insitu_volm_flow(qoil_std))
        vte = sp.velocity(qtot, ate)

        vte_ray = np.append(vte_ray, vte)
        mach_ray = np.append(mach_ray, vte / prop_su.cmix())

        pte_ray = np.append(pte_ray, pte)
        rho_ray = np.append(rho_ray, prop_su.rho_mix())

        kse_ray = np.append(kse_ray, enterance_ke(ken, vte))
        ese_ray = np.append(ese_ray, ese_ray[-1] + incremental_ee(pte_ray[-2:], rho_ray[-2:]))
        tee_ray = np.append(tee_ray, kse_ray[-1] + ese_ray[-1])

    if mach_ray[-1] >= 1:
        tee_fin = np.interp(1, mach_ray, tee_ray)  # find tee where mach = 1
    else:
        tee_fin = tee_ray[-1]

    return tee_fin, qoil_std, pte_ray, rho_ray, vte_ray, tee_ray  # type: ignore


def tee_positive_slope(pte_ray: np.ndarray, tee_ray: np.ndarray) -> list:
    """Throat Entry Equation with Positive Slope

    Only keeps the points along the TEE that have a positive slope. Numpy gradient
    function uses central distance theorem, so the output is the same length as input.

    Args:
        pte_ray (np array): Pressure Throat Entry Array, psig
        tee_ray (np array): Throat Entry Equation Array, ft2/s2

    Returns:
        mask (list?): Identify points where slope is positive
    """
    dtdp = np.gradient(tee_ray, pte_ray)  # uses central limit thm, so same size
    mask = dtdp >= 0  # only points where slope is greater than or equal to zero
    return mask


def tee_zero(
    pte_ray: np.ndarray, rho_ray: np.ndarray, vte_ray: np.ndarray, tee_ray: np.ndarray
) -> tuple[float, float, float]:
    """Throat Entry Parameters with zero TEE

    Calculate the throat entry pressure, density, and velocity where TEE crosses zero.
    Valid for one suction pressure  of the pump / reservoir.

    Args:
        pte_ray (np array): Press Throat Entry Array, psig
        rho_ray (np array): Density Throat Entry Array, lbm/ft3
        vte_ray (np array): Velocity Throat Entry Array, ft/s
        tee_ray (np array): Throat Entry Equation Array, ft2/s2

    Return:
        pte (float): Throat Entry Pressure, psig
        rho_te (float): Throat Entry Density, lbm/ft3
        vte (float): Throat Entry Velocity, ft/s
    """
    mask = tee_positive_slope(pte_ray, tee_ray)  # only look at points with a positive slope

    pte = np.interp(0, np.flip(tee_ray[mask]), np.flip(pte_ray[mask]))
    rho_te = np.interp(0, np.flip(tee_ray[mask]), np.flip(rho_ray[mask]))
    vte = np.interp(0, np.flip(tee_ray[mask]), np.flip(vte_ray[mask]))

    return pte, rho_te, vte  # type: ignore


def psu_secant(psu1: float, psu2: float, dete1: float, dete2: float) -> float:
    """Next Suction Pressure with Secant Method

    Uses the secant method to calculate the next psu to use to find a zero dEte at Ma = 1.

    Args:
        psu1 (float): Suction Pressure One, psig
        psu2 (float): Suction Pressure Two, psig
        dete1 (float): Differential Energy at psu1 and Ma = 1, ft2/s2
        dete2 (float): Differential Energy at psu1 and Ma = 1, ft2/s2

    Return:
        psu3 (float): Suction Pressure Three, psig
    """
    psu3 = psu2 - dete2 * (psu1 - psu2) / (dete1 - dete2)
    return psu3


# really be called psu minimizer, because that is truely what is trying to be done
# cut this down so it just finds the minimum psu required?
def tee_minimize(
    tsu: float, ken: float, ate: float, ipr_su: InFlow, prop_su: ResMix
) -> tuple[float, float, float, float, float]:
    """Minimize Throat Entry Equation at pmo

    Find that psu that minimizes the throat entry equation for where Mach = 1 (pmo).
    Secant method for iteration, starting point is Reservoir Pressure minus 300 and 400 psig.

    Args:
        tsu (float): Suction Temp, deg F
        ken (float): Throat Entry Friction, unitless
        ate (float): Throat Entry Area, ft2
        ipr_su (InFlow): IPR of Reservoir
        prop_su (ResMix): Properties of Suction Fluid

    Returns:
        psu (float): Suction Pressure, psig
        qoil_std (float): Oil Rate, STBOPD
        pte (float): Throat Entry Pressure, psig
        rho_te (float): Throat Entry Density, lbm/ft3
        vte (float): Throat Entry Velocity, ft/s
    """
    psu_list = [ipr_su.pres - 300, ipr_su.pres - 400]
    # store values of tee near mach=1 pressure
    tee_list = [
        tee_last(psu_list[0], tsu, ken, ate, ipr_su, prop_su)[0],
        tee_last(psu_list[1], tsu, ken, ate, ipr_su, prop_su)[0],
    ]
    psu_diff = 5  # criteria for when you've converged to an answer
    n = 0  # loop counter
    while abs(psu_list[-2] - psu_list[-1]) > psu_diff:
        psu_nxt = psu_secant(psu_list[-2], psu_list[-1], tee_list[-2], tee_list[-1])
        tee_nxt, qoil_std, pte_ray, rho_ray, vte_ray, tee_ray = tee_last(psu_nxt, tsu, ken, ate, ipr_su, prop_su)
        psu_list.append(psu_nxt)
        tee_list.append(tee_nxt)
        n = n + 1
        if n == 10:
            print("TEE Minimization did not converge")
            break
    pte, rho_te, vte = tee_zero(pte_ray, rho_ray, vte_ray, tee_ray)  # type: ignore
    return psu_list[-1], qoil_std, pte, rho_te, vte  # type: ignore


def nozzle_velocity(pni: float, pte: float, knz: float, rho_nz: float) -> float:
    """Nozzle Velocity

    Solve Bernoulli's Equation to calculate the nozzle velocity in ft/s.

    Args:
        pni (float): Nozzle Inlet Pressure, psig
        pte (float): Throat Entry Pressure, psig
        knz (float): Friction of Nozzle, unitless
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3

    Returns:
        vnz (float): Nozzle Velocity, ft/s
    """
    vnz = math.sqrt(2 * 32.174 * 144 * (pni - pte) / (rho_nz * (1 + knz)))
    return vnz


def nozzle_rate(vnz: float, anz: float) -> tuple[float, float]:
    """Nozzle Flow Rate

    Find Nozzle / Power Fluid Flowrate in ft3/s and BPD

    Args:
        vnz (float): Nozzle Velocity, ft/s
        anz (float): Area of Nozzle, ft2

    Returns:
        qnz_ft3s (float): Nozzle Flowrate ft3/s
        qnz_bpd (float): Nozzle Flowrate bpd
    """
    qnz_ft3s = anz * vnz
    qnz_bpd = sp.ft3s_to_bpd(qnz_ft3s)
    return qnz_ft3s, qnz_bpd


def throat_inlet_momentum(
    vnz: float, anz: float, rho_nz: float, vte: float, ate: float, rho_te: float
) -> tuple[float, float]:
    """Throat Inlet Momentum

    Calculate the inlet momentum of the throat in lbm/(s2*ft). The units
    are actually pressure, but referred to as momentum to help differentiate.

    Args:
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3

    Returns:
        mom_nz (float): Nozzle Momentum, lbm*ft/s2
        mom_te (float): Entry Momentum, lbm*ft/s2
    """
    mom_nz = sp.momentum(rho_nz, vnz, anz)  # momentum of the nozzle
    mom_te = sp.momentum(rho_te, vte, ate)  # momentum of the throat entry
    return mom_nz, mom_te


def throat_outlet_momentum(kth: float, vtm: float, ath: float, rho_tm: float) -> tuple[float, float]:
    """Throat Outlet Momentum

    Calculate the outlet momentum of the throat in lbm/(s2*ft). The units
    are actually pressure, but referred to as momentum to help differentiate.
    Writing the equation this way exposes the somewhat arbitrary feeling of the 0.5 assigned to the friction.
    It is being considered whether to drop the 0.5 or keep it.

    Args:
        kth (float): Throat Friction Factor, unitless
        vtm (float): Velocity of Throat Mixture, ft/s
        ath (float): Area of the Throat, ft2
        rho_tm (float): Density of Throat Mixture, lbm/ft3

    Returns:
        mom_tm (float): Throat Mixting Momentum, lbm*ft/s2
        mom_fr (float): Throat Friction Momemum, lbm*ft/s2
    """
    mom_tm = sp.momentum(rho_tm, vtm, ath)
    mom_fr = 1 / 2 * kth * mom_tm
    return mom_tm, mom_fr


def throat_discharge(
    pte: float,
    tte: float,
    kth: float,
    vnz: float,
    anz: float,
    rho_nz: float,
    vte: float,
    ate: float,
    rho_te: float,
    prop_tm: ResMix,
):
    """Throat Discharge Pressure

    Solves the throat mixture equation of the jet pump. Calculates throat differntial pressure.
    Use the throat entry pressure and differential pressure to calculate throat mix pressure.
    Account for the discharge pressure is greater than the inlet pressure. Loops through the
    calculated discharge pressure until a converged answer occurs.

    Args:
        pte (float): Pressure of Throat Entry, psig
        tte (float): Temperature of Throat Entry, deg F
        kth (float): Friction of Throat Mix, Unitless
        vnz (float): Velocity of Nozzle, ft/s
        anz (float): Area of Nozzle, ft2
        rho_nz (float): Density of Nozzle Fluid, lbm/ft3
        vte (float): Velocity of Throat Entry Mixture, ft/s
        ate (float): Area of Throat Entry, ft2
        rho_te (float): Density of Throat Entry Mixture, lbm/ft3
        prop_tm (ResMix): Properties of the Throat Mixture

    Returns:
        ptm (float): Throat Discharge Pressure, psig
    """
    mom_nz, mom_te = throat_inlet_momentum(vnz, anz, rho_nz, vte, ate, rho_te)

    mnz = sp.massflow(rho_nz, vnz, anz)  # mass flow of the nozzle
    mte = sp.massflow(rho_te, vte, ate)  # mass flow of the throat entry
    ath = anz + ate  # area of the throat
    mtm = mnz + mte  # mass flow of total mixture

    rho_tm = prop_tm.condition(pte, tte).rho_mix()  # density of total mixture
    vtm = sp.velocity(mtm / rho_tm, ath)

    mom_tm, mom_fr = throat_outlet_momentum(kth, vtm, ath, rho_tm)
    mom_tot = mom_fr + mom_tm - mom_nz - mom_te
    dp_tm = sp.mom_to_psi(mom_tot, ath)  # lbf/in2

    ptm = pte - dp_tm
    ptm_list = [pte, ptm]

    ptm_diff = 5
    n = 0
    while abs(ptm_list[-2] - ptm_list[-1]) > ptm_diff:
        ptm = ptm_list[-1]

        rho_tm = prop_tm.condition(ptm, tte).rho_mix()  # density of total mixture
        vtm = sp.velocity(mtm / rho_tm, ath)

        mom_tm, mom_fr = throat_outlet_momentum(kth, vtm, ath, rho_tm)
        mom_tot = mom_fr + mom_tm - mom_nz - mom_te
        dp_tm = sp.mom_to_psi(mom_tot, ath)  # lbf/in2
        ptm = pte - dp_tm
        ptm_list.append(ptm)
        n = n + 1
        if n == 10:
            print("Throat Mixture did not converge")
            break
    # print(f"Throat took {n} loops to find solution")
    return ptm_list[-1]


def throat_wc(qoil_std: float, wc_su: float, qwat_nz: float) -> float:
    """Throat Watercut

    Calculate watercut inside jet pump throat. This is the new watercut
    after the power fluid and reservoir fluid have mixed together

    Args:
        qoil_std (float): Oil Rate, STD BOPD
        wc_su (float): Watercut at pump suction, decimal
        qwat_nz (float): Powerfluid Flowrate, BWPD

    Returns:
        wc_tm (float): Watercut at throat, decimal"""

    qwat_su = qoil_std * wc_su / (1 - wc_su)
    qwat_tot = qwat_nz + qwat_su
    wc_tm = qwat_tot / (qwat_tot + qoil_std)
    return wc_tm


def diffuser_ke(kdi: float, vtm: float, vdi: float) -> float:
    """Diffuser Kinetic Energy

    Calculate the kinetic energy in the diffuser.
    Substract the energy lost due to friction from inlet fluid.

    Args:
        kdi: Diffuser Friction Factor, unitless
        vtm: Velocity at Throat Mixture, ft/s
        vdi: Velocity at Diffuser Discharge, ft/s

    Returns:
        ke_di: Diffuser Kinetic Energy, ft2/s2
    """
    ke_di = (vdi**2 - (1 - kdi) * vtm**2) / 2
    return ke_di


def diffuser_discharge(
    ptm: float, ttm: float, kdi: float, ath: float, adi: float, qoil_std: float, prop_tm: ResMix
) -> tuple[float, float]:
    """Diffuser Discharge Pressure

    Directly calculate the diffuser discharge pressure. Only loops until the diffuser total
    energy is greater than 0. Then uses numpy interpolation for diffuser discharge pressure.

    Args:
        ptm (float): Throat Mixture Pressure, psig
        ttm (float): Throat Mixture Temp, deg F
        kdi (float): Diffuser Friction Factor, unitless
        ath (float): Throat Area, ft2
        adi (float): Diffuser / Tubing Area, ft2
        qoil_std (float): Oil Rate, STD BOPD
        prop_tm (ResMix): Properties of Throat Mixture

    Returns:
        vtm (float): Throat Mixture Velocity
        pdi (float): Diffuser Discharge Pressure, psig
    """
    prop_tm = prop_tm.condition(ptm, ttm)
    qtot = sum(prop_tm.insitu_volm_flow(qoil_std))

    vtm = sp.velocity(qtot, ath)
    vdi = sp.velocity(qtot, adi)

    pdi_ray = np.array([ptm])
    vdi_ray = np.array([vdi])
    rho_ray = np.array([prop_tm.rho_mix()])

    kse_ray = np.array([diffuser_ke(kdi, vtm, vdi)])
    ese_ray = np.array([0])
    dte_ray = np.array([kse_ray + ese_ray])

    n = 0
    pinc = 100  # pressure increase

    while dte_ray[-1] < 0:
        pdi = pdi_ray[-1] + pinc
        prop_tm = prop_tm.condition(pdi, ttm)
        qtot = sum(prop_tm.insitu_volm_flow(qoil_std))
        vdi = sp.velocity(qtot, adi)

        vdi_ray = np.append(vdi_ray, vdi)

        pdi_ray = np.append(pdi_ray, pdi)
        rho_ray = np.append(rho_ray, prop_tm.rho_mix())

        kse_ray = np.append(kse_ray, diffuser_ke(kdi, vtm, vdi))
        ese_ray = np.append(ese_ray, ese_ray[-1] + incremental_ee(pdi_ray[-2:], rho_ray[-2:]))
        dte_ray = np.append(dte_ray, kse_ray[-1] + ese_ray[-1])

        n = n + 1
        if n == 15:
            print("Diffuser did not find discharge pressure")
            break

    # print(f"Diffuser took {n} loops to find solution")
    pdi = np.interp(0, dte_ray, pdi_ray)
    return vtm, pdi  # type: ignore
