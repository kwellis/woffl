import json
from pathlib import Path

import numpy as np
import pytest

from woffl.pvt.blackoil import BlackOil


def compute_blackoil_data(
    prs_ray: np.ndarray | list, temp: float, oil_api: float, bubblepoint: float, gas_sg: float
) -> dict:
    """Compute BlackOil Data

    Create a list of properties of a formgas. Can be used to compare to results obtained with hysys.
    Density and oil viscosity.
    """
    py_oil = BlackOil(oil_api=oil_api, bubblepoint=bubblepoint, gas_sg=gas_sg)
    rho_oil, visc_oil = [], []

    for prs in prs_ray:
        py_gas = py_oil.condition(prs, temp)

        rho_oil.append(py_gas.density)
        visc_oil.append(py_gas.viscosity())

    pyoil = {"rho_oil": rho_oil, "visc_oil": visc_oil}
    return pyoil


@pytest.fixture(scope="module")
def hysys_blackoil():
    hysys_path = Path(__file__).parents[1] / "data" / "hysys_blackoil_peng_rob.json"
    with open(hysys_path) as json_file:
        return json.load(json_file)


@pytest.fixture(scope="module")
def python_blackoil(hysys_blackoil):
    return compute_blackoil_data(
        hysys_blackoil["pres_psig"],
        hysys_blackoil["temp_degf"],
        hysys_blackoil["oil_api"],
        hysys_blackoil["bubblepoint"],
        hysys_blackoil["gas_sg"],
    )


def test_oil_density(hysys_blackoil, python_blackoil) -> None:
    np.testing.assert_allclose(hysys_blackoil["rho_oil"], python_blackoil["rho_oil"], rtol=0.05)


def test_oil_viscosity(hysys_blackoil, python_blackoil) -> None:
    # 75% error, why are we even testing...haha
    np.testing.assert_allclose(hysys_blackoil["visc_oil"], python_blackoil["visc_oil"], rtol=0.75)


def test_oil_tension() -> None:
    py_boil = BlackOil.test_oil()
    py_boil.condition(2500, 80)
    assert py_boil.tension() / 0.0000685 == pytest.approx(16.04, rel=0.01)  # dyne/cm
