import pytest

import woffl.flow.singlephase as sp
import woffl.flow.twophase as tp

# only works if the command python -m tests.flow_test is used

book_Nd = 41.34
book_Ngv = 9.29
book_Nlv = 6.02
book_l1, book_l2 = 1.53, 0.88


# example problem in Two Phase Flow in Pipes by Beggs / Brill (1988) pg. 3-31
def test_ros_l1_l2() -> None:
    calc_l1, calc_l2 = tp.ros_lp(book_Nd)
    assert calc_l1 == pytest.approx(book_l1, rel=0.05)
    assert calc_l2 == pytest.approx(book_l2, rel=0.05)


def test_ros_regime() -> None:
    calc_vpat = tp.ros_flow_pattern(book_Ngv, book_Nlv, book_Nd)
    assert calc_vpat == "slug"


# example problem in Two Phase Flow in Pipes by Beggs / Brill (1988) pg. 3-62
book_nslh = 0.393
book_NFr = 5.67
book_hpat = "intermittent"
book_tparm = 1  # this is really n/a since it is intermittent flow
book_ilh = 0.512
book_dhyd = 0.249 * 12  # inches
book_rho_liq = 56.6
book_rho_gas = 2.84
book_uliq = 18
book_ugas = 0.018
book_NRe = 8450
book_vsg = 4.09  # ft/s
book_vsl = 2.65  # ft/s
book_vmix = book_vsg + book_vsl
book_ff = 0.032  # book doesn't say what value they use for absolute roughness
book_yf = 1.5
book_sf = 0.37
book_ftp = 0.046
pipe_len = 100  # feet
book_fric = 3.12 * pipe_len / 144  # convert from psf/ft to psi
book_stat = 30.36 * pipe_len / 144  # convert from psf/ft to psi

# function is the same for slip vs no slip, just depends on what value for slip vs no slip you use
book_rho_mix = tp.density_slip(book_rho_liq, book_rho_gas, book_nslh)
book_rho_slip = tp.density_slip(book_rho_liq, book_rho_gas, book_ilh)
book_umix = tp.density_slip(book_uliq, book_ugas, book_nslh)


def test_froude_number() -> None:
    calc_NFr = tp.froude(book_vmix, book_dhyd)
    assert calc_NFr == pytest.approx(book_NFr, rel=0.01)


def test_beggs_regime() -> None:
    calc_hpat, calc_tparm = tp.beggs_flow_pattern(book_nslh, book_NFr)
    assert calc_hpat == "intermittent"


def test_beggs_holdup() -> None:
    calc_ilh = tp.beggs_holdup_inc(book_nslh, book_NFr, book_Nlv, 90, book_hpat, book_tparm)
    assert calc_ilh == pytest.approx(book_ilh, rel=0.01)


def test_reynolds_number() -> None:
    calc_NRe = sp.reynolds(book_rho_mix, book_vmix, book_dhyd, book_umix)
    assert calc_NRe == pytest.approx(book_NRe, rel=0.01)


def test_friction_factor() -> None:
    calc_rr = sp.relative_roughness(book_dhyd, 0.004)  # book doesn't say what abs ruff they use...
    calc_ff = sp.ffactor_darcy(book_NRe, calc_rr)
    assert calc_ff == pytest.approx(book_ff, abs=0.03)


def test_beggs_yf() -> None:
    calc_yf = tp.beggs_yf(book_nslh, book_ilh)
    assert calc_yf == pytest.approx(book_yf, rel=0.01)


def test_beggs_sf() -> None:
    calc_sf = tp.beggs_sf(book_yf)
    assert calc_sf == pytest.approx(book_sf, rel=0.01)


def test_beggs_friction_factor() -> None:
    calc_ftp = tp.beggs_ff(book_ff, book_sf)
    assert calc_ftp == pytest.approx(book_ftp, rel=0.01)


def test_beggs_press_friction() -> None:
    calc_fric = tp.beggs_press_friction(book_ftp, book_rho_mix, book_vmix, book_dhyd, pipe_len)
    assert calc_fric == pytest.approx(book_fric, rel=0.01)


def test_beggs_press_static() -> None:
    calc_stat = tp.beggs_press_static(book_rho_slip, pipe_len)
    assert calc_stat == pytest.approx(book_stat, rel=0.01)
