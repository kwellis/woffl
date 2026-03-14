import woffl.flow.singlephase as sp
from woffl.geometry.pipe import PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.formwat import FormWater


def top_down_press(
    ptop: float, ttop: float, qwat_bpd: float, prop_pf: FormWater, annul: PipeInPipe, wellprof: WellProfile
) -> float:
    """Top Down Annulus Pressure Calculation

    Calculates the pressure increase in the annulus. This is the preferred method
    of calculation, as a negative pressure cannot be created during calculations.

    Args:
        ptop (float): Pressure at top node, psig
        ttop (float): Temperature at top node, deg F
        qwat_bpd (float): Power Fluid Rate, STBWPD
        prop_pf (FormWater): Properties of Power Fluid in Annulus, FormWater
        annul (Annulus): Annulus geometry inside the wellbore, Annulus
        wellprof (WellProfile): survey dimensions and location of jet pump, WellProfile

    Returns:
        pbot (list): Pressure in the annulus at the jetpump, psig
    """
    prop_pf = prop_pf.condition(ptop, ttop)  # doesn't do anything for water, but left for consistency

    qwat_cfs = sp.bpd_to_ft3s(qwat_bpd)
    vwat = sp.velocity(qwat_cfs, annul.ann_area)
    reyn = sp.reynolds(prop_pf.density, vwat, annul.hyd_dia, prop_pf.viscosity())
    rel_ruff = sp.relative_roughness(annul.hyd_dia, annul.out_pipe.abs_ruff)
    ff_darcy = sp.ffactor_darcy(reyn, rel_ruff)

    fric_dp = sp.diff_press_friction(ff_darcy, prop_pf.density, vwat, annul.hyd_dia, wellprof.jetpump_md)
    stat_dp = sp.diff_press_static(prop_pf.density, wellprof.jetpump_vd)
    pbot = ptop + stat_dp - fric_dp
    return pbot
