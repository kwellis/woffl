import pytest

from woffl.pvt import FormWater

# "test" water properties...
water = FormWater(wat_sg=1)


def test_water_density() -> None:
    assert water.density == pytest.approx(62.4)


def test_water_viscosity() -> None:
    assert water.viscosity == pytest.approx(0.75)


def test_water_compressibility() -> None:
    assert water.compress == pytest.approx(3.13e-6, rel=0.01)  # psi^-1


def test_water_tension() -> None:
    assert water.tension == pytest.approx(0.005, rel=0.01)  # lbf/ft
