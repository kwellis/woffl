"""Single Phase Flow Equations

Equations that are predominately used in single phase flows. These equations can
also be adapted and used in two phase flows, but require a simplification such as
no slip or homogenous mixing.
"""

import math


# unit conversion
def bpd_to_ft3s(q_bpd: float) -> float:
    """Convert liquid BPD to ft3/s

    Args:
        q_bpd (float): Volumetric Flow, BPD

    Returns:
        q_ft3s (float): Volumetric Flow, ft3/s
    """
    # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
    q_ft3s = q_bpd * 42 / (24 * 60 * 60 * 7.48052)
    return q_ft3s


# unit conversion
def ft3s_to_bpd(q_ft3s: float) -> float:
    """Convert liquid ft3/s to BPD

    Args:
        q_ft3s (float): Volumetric Flow, ft3/s

    Returns:
        q_bpd (float): Volumetric Flow, BPD
    """
    # 42 gal/bbl, 7.48052 gal/ft3, 24 hr/day, 60min/hour, 60sec/min
    q_bpd = q_ft3s * (24 * 60 * 60 * 7.48052) / 42
    return q_bpd


def cross_area(dia: float) -> float:
    """Cross Sectional Area

    Calculate cross sectional area of piece of piping.

    Args:
        dia (float): Diameter, inches

    Returns:
        area (float): Area, ft2
    """
    dia = dia / 12  # ft
    area = math.pi * (dia**2) / 4  # ft2
    return area


def mom_to_psi(mom: float, area: float) -> float:
    """Convert Momentum across an area to lbf/in2

    Args:
        mom (float): Fluid Momentum, lbm*ft/s2
        area (float): Cross Sectional Area, ft2

    Return:
        p_mom (float): Equivalent Pressure of Momentum, psi
    """
    p_mom = mom / (area * 32.174 * 144)
    return p_mom


def velocity(flow: float, area: float) -> float:
    """Velocity Single Phase

    Args:
        flow (float): Volumetric Flow, ft3/s
        area (float): Pipe Cross Sectional Area, ft2

    Returns:
        vel (float): Flow Velocity, ft/s
    """
    vel = flow / area
    return vel


def momentum(rho: float, vel: float, area: float) -> float:
    """Fluid Momentum

    Args:
        vel (float): Velocity of the Fluid, ft/s
        area (float): Cross Sectional Area of the Flow, ft2
        rho (float): Density of the Fluid, lbm/ft3

    Returns:
        mom_fld (float): Fluid Momentum, lbm*ft/s2
    """
    return rho * vel**2 * area


def massflow(rho: float, vel: float, area: float) -> float:
    """Fluid Mass Flow

    Args:
        vel (float): Velocity of the Fluid, ft/s
        area (float): Cross Sectional Area of the Flow, ft2
        rho (float): Density of the Fluid, lbm/ft3

    Returns:
        mflow (float): Fluid Mass Flow, lbm/s
    """
    return rho * vel * area


def reynolds(rho: float, vel: float, dhyd: float, visc: float) -> float:
    """Reynolds Number

    Args:
        rho (float): Fluid density, lbm/ft3
        vel (float): Flow velocity, ft/s
        dhyd (float): Hydraulic Diameter, inches
        visc (float): Dynamic Viscosity, cP

    Returns:
        reynolds (float): Reynolds number, unitless
    """
    # convert inch to feet
    dhyd = dhyd / 12  # feet

    # convert cp to lbm/(ft*s)
    visc = visc / 1488.2  # lbm/(ft*s)

    reynolds = rho * vel * dhyd / visc
    return reynolds


def relative_roughness(dhyd: float, abs_ruff: float) -> float:
    """Relative Rougness of a Piece of Pipe

    Args:
        dhyd (float): Hydraulic Diameter, inches
        abs_ruff (float): Absolute Roughness, inches

    Returns:
        rel_ruff (float): Relative Roughness, e/D unitless
    """
    return abs_ruff / dhyd


def serghide(reynolds: float, rel_ruff: float) -> float:
    """Serghide Equation for Darcy-Weisbach Friction Factor

    Args:
        reynolds (float): Reynolds Number, Re unitless.
        rel_ruff (float): Relative Roughness, e/D unitless

    Returns:
        ff (float): Darcy-Weisbach Piping Friction Factor, unitless

    References:
        - Cranes Technical Paper No. 410 Equation 1-21
    """
    a = -2 * math.log10((rel_ruff / 3.7) + (12 / reynolds))
    b = -2 * math.log10((rel_ruff / 3.7) + (2.51 * a / reynolds))
    c = -2 * math.log10((rel_ruff / 3.7) + (2.51 * b / reynolds))
    ff = (a - (b - a) ** 2 / (c - 2 * b + a)) ** -2
    return ff


def ffactor_darcy(reynolds: float, rel_ruff: float) -> float:
    """Friction Factor Darcy Weisbach for Piping

    Args:
        reynolds (float): Reynolds Number, Re unitless
        rel_ruff (float): Relative Roughness, e/D unitless

    Returns:
        ff (float): Darcy-Weisbach Friction Factor of Piping, unitless
    """
    # laminar / transistional flow
    if reynolds < 4000:
        ff = 64 / reynolds

    else:
        ff = serghide(reynolds, rel_ruff)
    return ff


# dp_friction, friction_dp, fric_dp, or dp_fric
def diff_press_friction(ff: float, rho: float, vel: float, dhyd: float, length: float) -> float:
    """Frictional Differential Pressure in Piping

    Calculate the frictional pressure loss in a piping system.
    Negative is calculation is against flow, positive is with the flow.
    Matters if doing bottom down or top up calculations.

    Args:
        ff (float): Darcy friction factor of the pipe, unitless
        rho (float): Density Fluid, lbm/ft3
        vel (float): Velocity Fluid, ft/s
        dhyd (float): Hydraulic Diameter, inches
        length (float): Length of Pipe Segment, feet

    Returns:
        dp_fric (float): Frictional Differential Pressure, psi
    """
    g = 32.174  # 1 lbf equals 32.174 lbm*ft/s2
    dhyd = dhyd / 12  # feet
    dp_fric = ff * rho * vel**2 * length / (2 * dhyd * g)  # lbf/ft2
    dp_fric = dp_fric / 144  # lbf/in2
    return dp_fric


# dp_static, static_dp, or stat_dp or dp_stat
def diff_press_static(rho: float, height: float) -> float:
    """Static or Gravity Differential Pressure

    Incompressible Fluid Static Differential Pressure.
    Positive height is up, negative height is down.

    Args:
        rho (float): Fluid Density, lbm/ft3
        height (float): Fluid Vertical Height, feet

    Returns:
        dp_stat (float): Static Differential Pressure, psi
    """
    dp_stat = rho * height / 144  # psi, gravity cancels each other out with US Units
    return dp_stat
