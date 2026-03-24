"""Tests for Single Phase Flow Equations

Test values derived from Crane Technical Paper No. 410 (TP410).
References to specific equations, tables, and examples are noted.

Crane TP410 Reference Properties Used:
    Water at 60 deg F (Table A-6):
        rho = 62.37 lbm/ft3
        mu = 1.1 cP
    Commercial Steel Pipe Roughness (Table A-23):
        e = 0.0018 inches
    Schedule 40 Pipe IDs (Table B-14):
        2" Sch 40: ID = 2.067 inches
        3" Sch 40: ID = 3.068 inches
        4" Sch 40: ID = 4.026 inches
        6" Sch 40: ID = 6.065 inches
"""

import math

import pytest

import woffl.flow.singlephase as sp
from woffl.geometry.pipe import Pipe, PipeInPipe


# ---------------------------------------------------------------------------
# Unit Conversions: bpd_to_ft3s and ft3s_to_bpd
# 1 bbl = 42 US gallons, 1 ft3 = 7.48052 gallons, 1 day = 86400 seconds
# ---------------------------------------------------------------------------
class TestUnitConversions:
    """Unit conversions between barrels per day and ft3/s."""

    def test_bpd_to_ft3s_known_value(self):
        """1 BPD = 42 gal/day = 42/(7.48052*86400) ft3/s."""
        result = sp.bpd_to_ft3s(1.0)
        expected = 42 / (7.48052 * 86400)
        assert result == pytest.approx(expected, rel=1e-4)

    def test_ft3s_to_bpd_known_value(self):
        """Inverse of above: 6.4984e-5 ft3/s = 1 BPD."""
        q_ft3s = 42 / (7.48052 * 86400)
        result = sp.ft3s_to_bpd(q_ft3s)
        assert result == pytest.approx(1.0, rel=1e-4)

    def test_bpd_ft3s_roundtrip(self):
        """Round-trip conversion should return original value."""
        q_bpd = 5000.0
        assert sp.ft3s_to_bpd(sp.bpd_to_ft3s(q_bpd)) == pytest.approx(q_bpd, rel=1e-10)

    def test_bpd_to_ft3s_1000bpd(self):
        """1000 BPD = 0.06498 ft3/s."""
        result = sp.bpd_to_ft3s(1000.0)
        expected = 1000 * 42 / (7.48052 * 86400)
        assert result == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Cross Sectional Area
# A = pi/4 * (D/12)^2 where D is in inches, result in ft2
# Pipe IDs from Crane TP410 Table B-14
# ---------------------------------------------------------------------------
class TestCrossArea:
    """Cross sectional area calculations for standard pipe sizes."""

    def test_cross_area_2in_sch40(self):
        """2" Sch 40 pipe, ID = 2.067". A = pi/4 * (2.067/12)^2 = 0.02330 ft2."""
        result = sp.cross_area(2.067)
        expected = math.pi / 4 * (2.067 / 12) ** 2
        assert result == pytest.approx(expected, rel=1e-4)

    def test_cross_area_4in_sch40(self):
        """4" Sch 40 pipe, ID = 4.026". A = pi/4 * (4.026/12)^2 = 0.08840 ft2."""
        result = sp.cross_area(4.026)
        expected = math.pi / 4 * (4.026 / 12) ** 2
        assert result == pytest.approx(expected, rel=1e-4)

    def test_cross_area_6in_sch40(self):
        """6" Sch 40 pipe, ID = 6.065". A = pi/4 * (6.065/12)^2 = 0.2006 ft2."""
        result = sp.cross_area(6.065)
        expected = math.pi / 4 * (6.065 / 12) ** 2
        assert result == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Velocity: V = Q / A (Crane TP410 Eq. 1-1)
# ---------------------------------------------------------------------------
class TestVelocity:
    """Flow velocity from volumetric flow and area."""

    def test_velocity_4in_pipe_400gpm(self):
        """Water at 400 gpm through 4" Sch 40 pipe.

        Q = 400 gal/min * ft3/7.48052 gal * min/60 s = 0.8912 ft3/s
        A = pi/4 * (4.026/12)^2 = 0.08840 ft2
        V = Q/A = 10.08 ft/s
        """
        q_ft3s = 400 / (7.48052 * 60)
        area = math.pi / 4 * (4.026 / 12) ** 2
        result = sp.velocity(q_ft3s, area)
        expected = q_ft3s / area
        assert result == pytest.approx(expected, rel=1e-4)
        assert result == pytest.approx(10.08, rel=0.01)

    def test_velocity_2in_pipe_50gpm(self):
        """Water at 50 gpm through 2" Sch 40 pipe.

        Q = 50/(7.48052*60) = 0.1114 ft3/s
        A = pi/4 * (2.067/12)^2 = 0.02330 ft2
        V = 4.78 ft/s
        """
        q_ft3s = 50 / (7.48052 * 60)
        area = math.pi / 4 * (2.067 / 12) ** 2
        result = sp.velocity(q_ft3s, area)
        assert result == pytest.approx(4.78, rel=0.01)


# ---------------------------------------------------------------------------
# Mass Flow: mdot = rho * V * A
# ---------------------------------------------------------------------------
class TestMassFlow:
    """Mass flow rate from density, velocity, and area."""

    def test_massflow_water_4in(self):
        """Water at 60 deg F, 400 gpm, 4" Sch 40 pipe.

        rho = 62.37 lbm/ft3
        V = 10.08 ft/s
        A = 0.08840 ft2
        mdot = 62.37 * 10.08 * 0.08840 = 55.59 lbm/s
        """
        rho = 62.37
        area = math.pi / 4 * (4.026 / 12) ** 2
        q_ft3s = 400 / (7.48052 * 60)
        vel = q_ft3s / area
        result = sp.massflow(rho, vel, area)
        expected = rho * vel * area  # = rho * q_ft3s
        assert result == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Momentum: M = rho * V^2 * A
# ---------------------------------------------------------------------------
class TestMomentum:
    """Fluid momentum calculations."""

    def test_momentum_water_4in(self):
        """Water at 60 deg F, 400 gpm, 4" Sch 40 pipe.

        M = 62.37 * 10.08^2 * 0.08840 = 560.3 lbm*ft/s2
        """
        rho = 62.37
        area = math.pi / 4 * (4.026 / 12) ** 2
        q_ft3s = 400 / (7.48052 * 60)
        vel = q_ft3s / area
        result = sp.momentum(rho, vel, area)
        expected = rho * vel**2 * area
        assert result == pytest.approx(expected, rel=1e-4)

    def test_mom_to_psi_conversion(self):
        """Convert momentum to equivalent pressure.

        p = M / (A * gc * 144)
        For M = 560 lbm*ft/s2, A = 0.08840 ft2:
        p = 560 / (0.08840 * 32.174 * 144) = 1.367 psi
        """
        mom = 560.0
        area = 0.08840
        result = sp.mom_to_psi(mom, area)
        expected = 560.0 / (0.08840 * 32.174 * 144)
        assert result == pytest.approx(expected, rel=1e-3)


# ---------------------------------------------------------------------------
# Reynolds Number (Crane TP410 Eq. 1-2)
# Re = rho * V * D / mu
# Function takes: rho (lbm/ft3), V (ft/s), D (inches), mu (cP)
# Internal conversions: D/12 for ft, mu/1488.2 for lbm/(ft*s)
# ---------------------------------------------------------------------------
class TestReynolds:
    """Reynolds number calculations per Crane TP410 Eq. 1-2."""

    def test_reynolds_water_4in_sch40(self):
        """Water at 60 deg F, 400 gpm, 4" Sch 40 pipe (ID = 4.026").

        rho = 62.37 lbm/ft3, mu = 1.1 cP, V = 10.08 ft/s
        Re = 62.37 * 10.08 * (4.026/12) / (1.1/1488.2) = 285,400
        Crane TP410 would classify this as turbulent flow (Re > 4000).
        """
        rho = 62.37
        area = math.pi / 4 * (4.026 / 12) ** 2
        q_ft3s = 400 / (7.48052 * 60)
        vel = q_ft3s / area
        result = sp.reynolds(rho, vel, 4.026, 1.1)
        expected = rho * vel * (4.026 / 12) / (1.1 / 1488.2)
        assert result == pytest.approx(expected, rel=1e-4)
        assert result > 4000  # turbulent

    def test_reynolds_water_2in_sch40(self):
        """Water at 60 deg F, 50 gpm, 2" Sch 40 pipe (ID = 2.067").

        V = 4.78 ft/s, Re = 82,700. Turbulent.
        """
        rho = 62.37
        area = math.pi / 4 * (2.067 / 12) ** 2
        q_ft3s = 50 / (7.48052 * 60)
        vel = q_ft3s / area
        result = sp.reynolds(rho, vel, 2.067, 1.1)
        expected = rho * vel * (2.067 / 12) / (1.1 / 1488.2)
        assert result == pytest.approx(expected, rel=1e-4)
        assert result > 4000

    def test_reynolds_laminar(self):
        """Low velocity, high viscosity fluid -> laminar flow.

        Heavy oil: rho = 55 lbm/ft3, mu = 100 cP, V = 0.1 ft/s, D = 2.067"
        Re = 55 * 0.1 * (2.067/12) / (100/1488.2) = 13.7
        """
        result = sp.reynolds(55.0, 0.1, 2.067, 100.0)
        expected = 55.0 * 0.1 * (2.067 / 12) / (100.0 / 1488.2)
        assert result == pytest.approx(expected, rel=1e-4)
        assert result < 2100  # laminar

    def test_reynolds_water_6in_sch40_1000gpm(self):
        """Water at 60 deg F, 1000 gpm, 6" Sch 40 pipe (ID = 6.065").

        Q = 1000/(7.48052*60) = 2.228 ft3/s
        A = pi/4 * (6.065/12)^2 = 0.2006 ft2
        V = 11.11 ft/s
        Re = 62.37 * 11.11 * (6.065/12) / (1.1/1488.2) = 473,000
        """
        rho = 62.37
        area = math.pi / 4 * (6.065 / 12) ** 2
        q_ft3s = 1000 / (7.48052 * 60)
        vel = q_ft3s / area
        result = sp.reynolds(rho, vel, 6.065, 1.1)
        expected = rho * vel * (6.065 / 12) / (1.1 / 1488.2)
        assert result == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Relative Roughness (Crane TP410 Table A-23)
# e/D where e is absolute roughness, D is hydraulic diameter
# Commercial steel: e = 0.0018 inches
# ---------------------------------------------------------------------------
class TestRelativeRoughness:
    """Relative roughness per Crane TP410 Table A-23."""

    def test_rel_rough_4in_commercial_steel(self):
        """4" Sch 40 commercial steel: e/D = 0.0018/4.026 = 0.000447."""
        result = sp.relative_roughness(4.026, 0.0018)
        assert result == pytest.approx(0.000447, rel=0.01)

    def test_rel_rough_2in_commercial_steel(self):
        """2" Sch 40 commercial steel: e/D = 0.0018/2.067 = 0.000871."""
        result = sp.relative_roughness(2.067, 0.0018)
        assert result == pytest.approx(0.000871, rel=0.01)

    def test_rel_rough_6in_commercial_steel(self):
        """6" Sch 40 commercial steel: e/D = 0.0018/6.065 = 0.000297."""
        result = sp.relative_roughness(6.065, 0.0018)
        assert result == pytest.approx(0.000297, rel=0.01)


# ---------------------------------------------------------------------------
# Darcy Friction Factor (Crane TP410 Eq. 1-4 laminar, Eq. 1-21 Serghide)
#
# Laminar: f = 64/Re (Re < 4000)
# Turbulent: Serghide explicit solution to Colebrook (Eq. 1-21)
#
# Crane TP410 Moody Chart (Figure 1-3) benchmark values for commercial steel:
#   Re = 100,000, e/D = 0.0004 -> f = 0.0199
#   Re = 1,000,000, e/D = 0.0004 -> f = 0.0165
#   Re = 100,000, e/D = 0.001 -> f = 0.0222
# ---------------------------------------------------------------------------
class TestFrictionFactor:
    """Darcy-Weisbach friction factor per Crane TP410."""

    def test_laminar_re_1000(self):
        """Laminar flow: f = 64/Re. At Re = 1000, f = 0.064."""
        result = sp.ffactor_darcy(1000.0, 0.001)
        assert result == pytest.approx(0.064, rel=1e-6)

    def test_laminar_re_500(self):
        """Laminar flow: f = 64/500 = 0.128."""
        result = sp.ffactor_darcy(500.0, 0.001)
        assert result == pytest.approx(0.128, rel=1e-6)

    def test_laminar_re_2000(self):
        """Laminar flow: f = 64/2000 = 0.032."""
        result = sp.ffactor_darcy(2000.0, 0.001)
        assert result == pytest.approx(0.032, rel=1e-6)

    def test_laminar_independent_of_roughness(self):
        """In laminar flow, friction factor is independent of pipe roughness."""
        f1 = sp.ffactor_darcy(1000.0, 0.0001)
        f2 = sp.ffactor_darcy(1000.0, 0.001)
        f3 = sp.ffactor_darcy(1000.0, 0.01)
        assert f1 == pytest.approx(f2, rel=1e-6)
        assert f2 == pytest.approx(f3, rel=1e-6)

    def test_turbulent_moody_chart_point1(self):
        """Moody chart (TP410 Fig. 1-3): Re=100,000, e/D=0.0004.

        Colebrook solution: f = 0.0199. Serghide matches Colebrook.
        """
        result = sp.ffactor_darcy(100_000, 0.0004)
        assert result == pytest.approx(0.0199, rel=0.01)

    def test_turbulent_moody_chart_point2(self):
        """Moody chart (TP410 Fig. 1-3): Re=1,000,000, e/D=0.0004 -> f = 0.0165."""
        result = sp.ffactor_darcy(1_000_000, 0.0004)
        assert result == pytest.approx(0.0165, rel=0.02)

    def test_turbulent_moody_chart_point3(self):
        """Moody chart (TP410 Fig. 1-3): Re=100,000, e/D=0.001.

        Colebrook solution: f = 0.0222. Serghide matches Colebrook.
        """
        result = sp.ffactor_darcy(100_000, 0.001)
        assert result == pytest.approx(0.0222, rel=0.01)

    def test_turbulent_smooth_pipe(self):
        """Smooth pipe (e/D near 0): f approaches Blasius at moderate Re.

        At Re=100,000, smooth pipe f = 0.0180 (Moody chart).
        """
        result = sp.ffactor_darcy(100_000, 0.00001)
        assert result == pytest.approx(0.0180, rel=0.02)

    def test_fully_rough_zone(self):
        """Fully rough zone: f independent of Re, depends only on e/D.

        At very high Re with e/D = 0.01, f = 0.0380 (Moody chart).
        """
        f_high = sp.ffactor_darcy(10_000_000, 0.01)
        f_higher = sp.ffactor_darcy(100_000_000, 0.01)
        # In fully rough zone, f should be nearly identical
        assert f_high == pytest.approx(f_higher, rel=0.005)
        assert f_high == pytest.approx(0.0380, rel=0.02)

    def test_serghide_matches_colebrook(self):
        """Serghide (TP410 Eq. 1-21) is an explicit approximation of Colebrook.

        Verify Serghide result matches iterative Colebrook solution.
        Re = 200,000, e/D = 0.0005.
        Colebrook: 1/sqrt(f) = -2*log10(e/D/3.7 + 2.51/(Re*sqrt(f)))
        """
        re = 200_000
        rel_ruff = 0.0005

        # Iterative Colebrook solution
        f_guess = 0.02
        for _ in range(50):
            rhs = -2 * math.log10(rel_ruff / 3.7 + 2.51 / (re * math.sqrt(f_guess)))
            f_guess = (1 / rhs) ** 2

        result = sp.serghide(re, rel_ruff)
        assert result == pytest.approx(f_guess, rel=1e-4)


# ---------------------------------------------------------------------------
# Frictional Pressure Drop (Crane TP410 Eq. 1-4, Darcy-Weisbach)
# dP = f * L * rho * V^2 / (2 * D * gc * 144)
# where gc = 32.174 lbm*ft/(lbf*s2), 144 converts lbf/ft2 to psi
#
# Crane TP410 Example 4-1 type: Water at 60 deg F through Sch 40 steel pipe
# ---------------------------------------------------------------------------
class TestDiffPressFriction:
    """Frictional pressure drop per Crane TP410 Darcy-Weisbach equation."""

    def test_dp_friction_water_4in_400gpm(self):
        """Water at 60 deg F, 400 gpm, 4" Sch 40, 100 ft, commercial steel.

        rho = 62.37 lbm/ft3, mu = 1.1 cP, ID = 4.026", e = 0.0018"
        V = 10.08 ft/s, Re = 285,400, e/D = 0.000447
        f = 0.0178 (Moody chart / Serghide)
        dP = 0.0178 * 62.37 * 10.08^2 * 100 / (2 * (4.026/12) * 32.174 * 144)
        dP = 3.6 psi per 100 ft (Crane TP410 typical result)
        """
        rho = 62.37
        dhyd = 4.026
        length = 100.0
        area = sp.cross_area(dhyd)
        q_ft3s = 400 / (7.48052 * 60)
        vel = sp.velocity(q_ft3s, area)
        re = sp.reynolds(rho, vel, dhyd, 1.1)
        rel_ruff = sp.relative_roughness(dhyd, 0.0018)
        ff = sp.ffactor_darcy(re, rel_ruff)
        result = sp.diff_press_friction(ff, rho, vel, dhyd, length)

        # Hand calculation cross-check
        expected = ff * rho * vel**2 * length / (2 * (dhyd / 12) * 32.174 * 144)
        assert result == pytest.approx(expected, rel=1e-4)
        # TP410 expected range for this scenario
        assert 3.0 < result < 4.5

    def test_dp_friction_water_2in_50gpm(self):
        """Water at 60 deg F, 50 gpm, 2" Sch 40, 100 ft, commercial steel.

        ID = 2.067", V = 4.78 ft/s, Re = 82,700
        Smaller pipe = higher friction loss per unit length at same velocity.
        """
        rho = 62.37
        dhyd = 2.067
        length = 100.0
        area = sp.cross_area(dhyd)
        q_ft3s = 50 / (7.48052 * 60)
        vel = sp.velocity(q_ft3s, area)
        re = sp.reynolds(rho, vel, dhyd, 1.1)
        rel_ruff = sp.relative_roughness(dhyd, 0.0018)
        ff = sp.ffactor_darcy(re, rel_ruff)
        result = sp.diff_press_friction(ff, rho, vel, dhyd, length)

        expected = ff * rho * vel**2 * length / (2 * (dhyd / 12) * 32.174 * 144)
        assert result == pytest.approx(expected, rel=1e-4)
        # Reasonable range for this scenario
        assert result > 0

    def test_dp_friction_scales_with_length(self):
        """Pressure drop should scale linearly with pipe length (TP410 Eq. 1-4)."""
        ff, rho, vel, dhyd = 0.02, 62.37, 10.0, 4.026
        dp_100 = sp.diff_press_friction(ff, rho, vel, dhyd, 100.0)
        dp_200 = sp.diff_press_friction(ff, rho, vel, dhyd, 200.0)
        assert dp_200 == pytest.approx(2 * dp_100, rel=1e-6)

    def test_dp_friction_scales_with_vel_squared(self):
        """Pressure drop should scale with V^2 (TP410 Eq. 1-4)."""
        ff, rho, dhyd, length = 0.02, 62.37, 4.026, 100.0
        dp_v5 = sp.diff_press_friction(ff, rho, 5.0, dhyd, length)
        dp_v10 = sp.diff_press_friction(ff, rho, 10.0, dhyd, length)
        assert dp_v10 == pytest.approx(4.0 * dp_v5, rel=1e-6)

    def test_dp_friction_water_6in_1000gpm(self):
        """Water at 60 deg F, 1000 gpm, 6" Sch 40, 100 ft.

        ID = 6.065", V = 11.11 ft/s
        Larger diameter = lower friction loss than 4" at similar velocity.
        """
        rho = 62.37
        dhyd = 6.065
        length = 100.0
        area = sp.cross_area(dhyd)
        q_ft3s = 1000 / (7.48052 * 60)
        vel = sp.velocity(q_ft3s, area)
        re = sp.reynolds(rho, vel, dhyd, 1.1)
        rel_ruff = sp.relative_roughness(dhyd, 0.0018)
        ff = sp.ffactor_darcy(re, rel_ruff)
        result = sp.diff_press_friction(ff, rho, vel, dhyd, length)

        expected = ff * rho * vel**2 * length / (2 * (dhyd / 12) * 32.174 * 144)
        assert result == pytest.approx(expected, rel=1e-4)


# ---------------------------------------------------------------------------
# Static / Hydrostatic Pressure (Crane TP410)
# dP = rho * h / 144
# For water at 60 deg F: 62.37 lbm/ft3
# Gradient = 0.433 psi/ft (well-known water column value)
# ---------------------------------------------------------------------------
class TestDiffPressStatic:
    """Hydrostatic pressure calculations."""

    def test_static_water_100ft(self):
        """100 ft water column at 60 deg F.

        dP = 62.37 * 100 / 144 = 43.31 psi
        Known value: ~0.433 psi/ft * 100 ft = 43.3 psi.
        """
        result = sp.diff_press_static(62.37, 100.0)
        assert result == pytest.approx(43.31, rel=0.01)

    def test_static_water_gradient(self):
        """Water pressure gradient = 0.433 psi/ft (Crane TP410 standard)."""
        gradient = sp.diff_press_static(62.37, 1.0)
        assert gradient == pytest.approx(0.433, rel=0.01)

    def test_static_scales_linearly(self):
        """Static pressure scales linearly with height."""
        dp_100 = sp.diff_press_static(62.37, 100.0)
        dp_200 = sp.diff_press_static(62.37, 200.0)
        assert dp_200 == pytest.approx(2 * dp_100, rel=1e-6)

    def test_static_negative_height(self):
        """Negative height gives negative (decreasing) pressure."""
        result = sp.diff_press_static(62.37, -100.0)
        assert result == pytest.approx(-43.31, rel=0.01)

    def test_static_brine(self):
        """Heavier fluid (brine = 72 lbm/ft3) gives higher gradient.

        dP = 72 * 100 / 144 = 50.0 psi per 100 ft.
        """
        result = sp.diff_press_static(72.0, 100.0)
        assert result == pytest.approx(50.0, rel=1e-4)


# ---------------------------------------------------------------------------
# Integration Test: Full Pipeline Pressure Drop Calculation
# Combines all functions as would be done in a Crane TP410 worked example
# ---------------------------------------------------------------------------
class TestIntegration:
    """End-to-end pipeline pressure drop combining all single phase functions."""

    def test_full_pipeline_calc_4in(self):
        """Complete Crane TP410-style pipeline calculation.

        Scenario: 400 gpm water at 60 deg F through 500 ft of 4" Sch 40
        commercial steel pipe with 200 ft of elevation gain.

        Total dP = dP_friction + dP_static
        """
        # Fluid properties (TP410 Table A-6, water at 60 deg F)
        rho = 62.37  # lbm/ft3
        visc = 1.1  # cP

        # Pipe properties (TP410 Table B-14, 4" Sch 40)
        dhyd = 4.026  # inches
        roughness = 0.0018  # inches, commercial steel (Table A-23)
        length = 500.0  # ft
        elevation = 200.0  # ft

        # Flow rate: 400 gpm converted to ft3/s
        q_ft3s = 400 / (7.48052 * 60)

        # Calculate
        area = sp.cross_area(dhyd)
        vel = sp.velocity(q_ft3s, area)
        re = sp.reynolds(rho, vel, dhyd, visc)
        rel_ruff = sp.relative_roughness(dhyd, roughness)
        ff = sp.ffactor_darcy(re, rel_ruff)
        dp_fric = sp.diff_press_friction(ff, rho, vel, dhyd, length)
        dp_stat = sp.diff_press_static(rho, elevation)

        total_dp = dp_fric + dp_stat

        # Verify individual components are reasonable
        assert re > 4000  # turbulent flow
        assert 0.015 < ff < 0.025  # reasonable Darcy friction factor
        assert dp_fric > 0  # friction always positive
        assert dp_stat > 0  # elevation gain = positive static head

        # Friction: ~3.6 psi/100ft * 5 = ~18 psi
        assert 15 < dp_fric < 25

        # Static: 62.37 * 200 / 144 = 86.6 psi
        assert dp_stat == pytest.approx(86.63, rel=0.01)

        # Total should be friction + static
        assert total_dp == pytest.approx(dp_fric + dp_stat, rel=1e-6)


# ---------------------------------------------------------------------------
# PipeInPipe Geometry Validation
# Verifies tubing and annulus geometric properties against hand calculations.
#
# 4.5" tubing (OD=4.5, wall=0.271) inside 7" casing (OD=7.0, wall=0.5):
#   Tubing ID = 4.5 - 2*0.271 = 3.958"
#   Casing ID = 7.0 - 2*0.5   = 6.0"
#   Annular Dhyd = 6.0 - 4.5  = 1.5"
# ---------------------------------------------------------------------------
class TestPipeInPipeGeometry:
    """Verify PipeInPipe geometric properties for standard oilfield pipe sizes."""

    @pytest.fixture
    def wellbore_4half_by_7(self):
        """4.5" tubing inside 7" casing."""
        tube = Pipe(out_dia=4.5, thick=0.271)
        case = Pipe(out_dia=7.0, thick=0.5)
        return PipeInPipe(inn_pipe=tube, out_pipe=case)

    @pytest.fixture
    def wellbore_3half_by_7(self):
        """3.5" tubing inside 7" casing."""
        tube = Pipe(out_dia=3.5, thick=0.271)
        case = Pipe(out_dia=7.0, thick=0.5)
        return PipeInPipe(inn_pipe=tube, out_pipe=case)

    def test_tubing_hyd_dia_4half(self, wellbore_4half_by_7):
        """Tubing Dhyd = tubing ID = 4.5 - 2*0.271 = 3.958"."""
        assert wellbore_4half_by_7.tube_hyd_dia == pytest.approx(3.958, rel=1e-4)

    def test_tubing_area_4half(self, wellbore_4half_by_7):
        """Tubing area = pi/4 * (3.958/12)^2 = 0.08553 ft2."""
        expected = math.pi / 4 * (3.958 / 12) ** 2
        assert wellbore_4half_by_7.tube_area == pytest.approx(expected, rel=1e-4)

    def test_annulus_hyd_dia_4half_by_7(self, wellbore_4half_by_7):
        """Annular Dhyd = casing ID - tubing OD = 6.0 - 4.5 = 1.5"."""
        assert wellbore_4half_by_7.ann_hyd_dia == pytest.approx(1.5, rel=1e-4)

    def test_annulus_area_4half_by_7(self, wellbore_4half_by_7):
        """Annular area = pi/4 * (6.0^2 - 4.5^2) / 144 = 0.08590 ft2."""
        expected = math.pi / 4 * (6.0**2 - 4.5**2) / 144
        assert wellbore_4half_by_7.ann_area == pytest.approx(expected, rel=1e-4)

    def test_annulus_hyd_dia_3half_by_7(self, wellbore_3half_by_7):
        """Annular Dhyd = 6.0 - 3.5 = 2.5"."""
        assert wellbore_3half_by_7.ann_hyd_dia == pytest.approx(2.5, rel=1e-4)

    def test_annulus_area_3half_by_7(self, wellbore_3half_by_7):
        """Annular area = pi/4 * (6.0^2 - 3.5^2) / 144 = 0.1295 ft2."""
        expected = math.pi / 4 * (6.0**2 - 3.5**2) / 144
        assert wellbore_3half_by_7.ann_area == pytest.approx(expected, rel=1e-4)

    def test_annulus_area_larger_than_zero(self, wellbore_4half_by_7):
        """Annular area must be positive."""
        assert wellbore_4half_by_7.ann_area > 0

    def test_inner_pipe_too_large_raises(self):
        """Inner pipe OD > outer pipe ID should raise ValueError."""
        big_tube = Pipe(out_dia=6.5, thick=0.3)
        small_case = Pipe(out_dia=7.0, thick=0.5)  # ID=6.0
        with pytest.raises(ValueError, match="not fit"):
            PipeInPipe(inn_pipe=big_tube, out_pipe=small_case)


# ---------------------------------------------------------------------------
# PipeInPipe Tubing Pressure Drop
# Validates that using PipeInPipe tubing properties with singlephase functions
# gives the same result as a direct pipe calculation. This confirms the tubing
# side is correctly wired and matches Crane TP410 methodology.
#
# Water at 60 deg F (TP410 Table A-6): rho = 62.37 lbm/ft3, mu = 1.1 cP
# 4.5" tubing: ID = 3.958", e = 0.004" (default roughness)
# ---------------------------------------------------------------------------
class TestPipeInPipeTubingPressureDrop:
    """Tubing pressure drop through PipeInPipe matches direct pipe calculation."""

    @pytest.fixture
    def wellbore(self):
        tube = Pipe(out_dia=4.5, thick=0.271)
        case = Pipe(out_dia=7.0, thick=0.5)
        return PipeInPipe(inn_pipe=tube, out_pipe=case)

    def test_tubing_dp_matches_direct_calc(self, wellbore):
        """Water at 60 deg F, 5000 BPD through 4.5" tubing, 1000 ft.

        PipeInPipe tubing properties should give identical results to
        using the tubing ID directly in singlephase functions.
        """
        rho = 62.37
        visc = 1.1
        length = 1000.0
        q_ft3s = sp.bpd_to_ft3s(5000.0)

        # Via PipeInPipe
        dhyd_pip = wellbore.tube_hyd_dia
        area_pip = wellbore.tube_area
        ruff_pip = wellbore.tube_abs_ruff
        vel_pip = sp.velocity(q_ft3s, area_pip)
        re_pip = sp.reynolds(rho, vel_pip, dhyd_pip, visc)
        rel_ruff_pip = sp.relative_roughness(dhyd_pip, ruff_pip)
        ff_pip = sp.ffactor_darcy(re_pip, rel_ruff_pip)
        dp_pip = sp.diff_press_friction(ff_pip, rho, vel_pip, dhyd_pip, length)

        # Direct calculation with tubing ID
        tube_id = 3.958  # 4.5 - 2*0.271
        area_dir = sp.cross_area(tube_id)
        vel_dir = sp.velocity(q_ft3s, area_dir)
        re_dir = sp.reynolds(rho, vel_dir, tube_id, visc)
        rel_ruff_dir = sp.relative_roughness(tube_id, 0.004)
        ff_dir = sp.ffactor_darcy(re_dir, rel_ruff_dir)
        dp_dir = sp.diff_press_friction(ff_dir, rho, vel_dir, tube_id, length)

        assert dp_pip == pytest.approx(dp_dir, rel=1e-6)

    def test_tubing_dp_reasonable_magnitude(self, wellbore):
        """5000 BPD water through 4.5" tubing, 5000 ft should give reasonable dp.

        At ~3.8 ft/s velocity in 3.958" tubing, friction is modest.
        """
        rho = 62.37
        visc = 1.1
        length = 5000.0
        q_ft3s = sp.bpd_to_ft3s(5000.0)

        vel = sp.velocity(q_ft3s, wellbore.tube_area)
        re = sp.reynolds(rho, vel, wellbore.tube_hyd_dia, visc)
        rel_ruff = sp.relative_roughness(wellbore.tube_hyd_dia, wellbore.tube_abs_ruff)
        ff = sp.ffactor_darcy(re, rel_ruff)
        dp_fric = sp.diff_press_friction(ff, rho, vel, wellbore.tube_hyd_dia, length)

        assert re > 4000  # turbulent
        assert dp_fric > 0
        # Friction dp through 5000 ft of tubing at modest rate should be
        # meaningful but not extreme
        assert 1 < dp_fric < 100


# ---------------------------------------------------------------------------
# PipeInPipe Annular Pressure Drop
# The annular hydraulic diameter (Dhyd = D_casing_ID - D_tubing_OD) is used
# with Darcy-Weisbach just like pipe flow. This is standard practice per
# API RP 13D and Bourgoyne et al. "Applied Drilling Engineering".
#
# Since Crane TP410 doesn't have annular flow examples, these tests validate:
#   1. Hand-calculated annular pressure drops match code output
#   2. Physical reasonableness (smaller annular gap = higher dp at same flow)
#   3. Darcy-Weisbach equation consistency with annular geometry
# ---------------------------------------------------------------------------
class TestPipeInPipeAnnularPressureDrop:
    """Annular pressure drop using PipeInPipe with singlephase functions."""

    @pytest.fixture
    def wb_4half_by_7(self):
        """4.5" tubing in 7" casing. Annular Dhyd = 1.5"."""
        tube = Pipe(out_dia=4.5, thick=0.271)
        case = Pipe(out_dia=7.0, thick=0.5)
        return PipeInPipe(inn_pipe=tube, out_pipe=case)

    @pytest.fixture
    def wb_3half_by_7(self):
        """3.5" tubing in 7" casing. Annular Dhyd = 2.5"."""
        tube = Pipe(out_dia=3.5, thick=0.271)
        case = Pipe(out_dia=7.0, thick=0.5)
        return PipeInPipe(inn_pipe=tube, out_pipe=case)

    def test_annular_dp_hand_calc(self, wb_4half_by_7):
        """Water at 60 deg F, 5000 BPD through 4.5 by 7" annulus, 1000 ft.

        Hand calculation using annular Dhyd = 1.5" in Darcy-Weisbach:
            A_ann = pi/4 * (6.0^2 - 4.5^2) / 144 = 0.08590 ft2
            Q = 5000 BPD -> ft3/s
            V = Q / A_ann
            Re = rho*V*Dhyd/mu
            f from Serghide
            dP = f * rho * V^2 * L / (2 * Dhyd * gc * 144)
        """
        rho = 62.37
        visc = 1.1
        length = 1000.0
        q_ft3s = sp.bpd_to_ft3s(5000.0)

        dhyd = wb_4half_by_7.ann_hyd_dia  # 1.5"
        area = wb_4half_by_7.ann_area
        abs_ruff = wb_4half_by_7.ann_abs_ruff

        vel = sp.velocity(q_ft3s, area)
        re = sp.reynolds(rho, vel, dhyd, visc)
        rel_ruff = sp.relative_roughness(dhyd, abs_ruff)
        ff = sp.ffactor_darcy(re, rel_ruff)
        dp_code = sp.diff_press_friction(ff, rho, vel, dhyd, length)

        # Hand calculation
        expected_area = math.pi / 4 * (6.0**2 - 4.5**2) / 144
        expected_vel = q_ft3s / expected_area
        expected_re = rho * expected_vel * (1.5 / 12) / (visc / 1488.2)
        expected_rr = abs_ruff / 1.5
        expected_ff = sp.serghide(expected_re, expected_rr)
        expected_dp = expected_ff * rho * expected_vel**2 * length / (2 * (1.5 / 12) * 32.174 * 144)

        assert dp_code == pytest.approx(expected_dp, rel=1e-4)

    def test_annular_dp_smaller_gap_higher_friction(self, wb_4half_by_7, wb_3half_by_7):
        """At the same flow rate, a tighter annulus (4.5 by 7") should have
        higher friction than a wider annulus (3.5 by 7") due to smaller Dhyd
        and higher velocity.
        """
        rho = 62.37
        visc = 1.1
        length = 1000.0
        q_ft3s = sp.bpd_to_ft3s(5000.0)

        # 4.5 by 7 annulus (tight, Dhyd=1.5")
        vel_tight = sp.velocity(q_ft3s, wb_4half_by_7.ann_area)
        re_tight = sp.reynolds(rho, vel_tight, wb_4half_by_7.ann_hyd_dia, visc)
        rr_tight = sp.relative_roughness(wb_4half_by_7.ann_hyd_dia, wb_4half_by_7.ann_abs_ruff)
        ff_tight = sp.ffactor_darcy(re_tight, rr_tight)
        dp_tight = sp.diff_press_friction(ff_tight, rho, vel_tight, wb_4half_by_7.ann_hyd_dia, length)

        # 3.5 by 7 annulus (wide, Dhyd=2.5")
        vel_wide = sp.velocity(q_ft3s, wb_3half_by_7.ann_area)
        re_wide = sp.reynolds(rho, vel_wide, wb_3half_by_7.ann_hyd_dia, visc)
        rr_wide = sp.relative_roughness(wb_3half_by_7.ann_hyd_dia, wb_3half_by_7.ann_abs_ruff)
        ff_wide = sp.ffactor_darcy(re_wide, rr_wide)
        dp_wide = sp.diff_press_friction(ff_wide, rho, vel_wide, wb_3half_by_7.ann_hyd_dia, length)

        # Tighter annulus = higher friction loss
        assert dp_tight > dp_wide

    def test_annular_dp_scales_with_length(self, wb_4half_by_7):
        """Annular friction pressure drop scales linearly with length."""
        rho = 62.37
        visc = 1.1
        q_ft3s = sp.bpd_to_ft3s(5000.0)

        vel = sp.velocity(q_ft3s, wb_4half_by_7.ann_area)
        re = sp.reynolds(rho, vel, wb_4half_by_7.ann_hyd_dia, visc)
        rr = sp.relative_roughness(wb_4half_by_7.ann_hyd_dia, wb_4half_by_7.ann_abs_ruff)
        ff = sp.ffactor_darcy(re, rr)

        dp_1000 = sp.diff_press_friction(ff, rho, vel, wb_4half_by_7.ann_hyd_dia, 1000.0)
        dp_2000 = sp.diff_press_friction(ff, rho, vel, wb_4half_by_7.ann_hyd_dia, 2000.0)
        assert dp_2000 == pytest.approx(2 * dp_1000, rel=1e-6)

    def test_annular_dp_vs_tubing_dp(self, wb_4half_by_7):
        """At the same flow rate, annular dp should differ from tubing dp
        due to different hydraulic diameters and areas.

        The annular Dhyd (1.5") is much smaller than tubing Dhyd (3.958"),
        but areas are similar (~0.086 ft2), so velocity is similar.
        The smaller annular Dhyd drives higher friction per unit length.
        """
        rho = 62.37
        visc = 1.1
        length = 1000.0
        q_ft3s = sp.bpd_to_ft3s(5000.0)

        # Tubing dp
        vel_t = sp.velocity(q_ft3s, wb_4half_by_7.tube_area)
        re_t = sp.reynolds(rho, vel_t, wb_4half_by_7.tube_hyd_dia, visc)
        rr_t = sp.relative_roughness(wb_4half_by_7.tube_hyd_dia, wb_4half_by_7.tube_abs_ruff)
        ff_t = sp.ffactor_darcy(re_t, rr_t)
        dp_tube = sp.diff_press_friction(ff_t, rho, vel_t, wb_4half_by_7.tube_hyd_dia, length)

        # Annulus dp
        vel_a = sp.velocity(q_ft3s, wb_4half_by_7.ann_area)
        re_a = sp.reynolds(rho, vel_a, wb_4half_by_7.ann_hyd_dia, visc)
        rr_a = sp.relative_roughness(wb_4half_by_7.ann_hyd_dia, wb_4half_by_7.ann_abs_ruff)
        ff_a = sp.ffactor_darcy(re_a, rr_a)
        dp_ann = sp.diff_press_friction(ff_a, rho, vel_a, wb_4half_by_7.ann_hyd_dia, length)

        # Annular dp should be higher due to smaller hydraulic diameter
        assert dp_ann > dp_tube
        # Both should be positive and finite
        assert dp_tube > 0
        assert dp_ann > 0

    def test_annular_static_pressure(self, wb_4half_by_7):
        """Static pressure is independent of geometry, only depends on
        fluid density and height. Same result for tubing or annulus.
        """
        rho = 62.37
        height = 5000.0
        dp_stat = sp.diff_press_static(rho, height)
        # 62.37 * 5000 / 144 = 2165 psi
        assert dp_stat == pytest.approx(62.37 * 5000 / 144, rel=1e-4)

    def test_annular_total_dp_powerfluid_style(self, wb_4half_by_7):
        """Simulate a power fluid injection scenario (reverse circulating).

        Water at 60 deg F, 3000 BPD injected down annulus (4.5 by 7"), 5000 ft deep.
        Total pressure at jet pump = surface pressure + static - friction.
        This mirrors the powerfluid_top_down_press function logic.
        """
        rho = 62.37
        visc = 1.1
        p_surface = 500.0  # psi surface injection pressure
        depth = 5000.0  # ft TVD
        length = 5000.0  # ft MD (assume vertical)
        q_ft3s = sp.bpd_to_ft3s(3000.0)

        # Annular flow properties
        vel = sp.velocity(q_ft3s, wb_4half_by_7.ann_area)
        re = sp.reynolds(rho, vel, wb_4half_by_7.ann_hyd_dia, visc)
        rr = sp.relative_roughness(wb_4half_by_7.ann_hyd_dia, wb_4half_by_7.ann_abs_ruff)
        ff = sp.ffactor_darcy(re, rr)

        dp_fric = sp.diff_press_friction(ff, rho, vel, wb_4half_by_7.ann_hyd_dia, length)
        dp_stat = sp.diff_press_static(rho, depth)  # positive going down

        # Pressure at depth = surface + hydrostatic - friction
        p_bottom = p_surface + dp_stat - dp_fric

        assert dp_fric > 0
        assert dp_stat > 0
        # Hydrostatic gain should dominate over friction for this scenario
        assert dp_stat > dp_fric
        # Bottom hole pressure should be higher than surface pressure
        assert p_bottom > p_surface
