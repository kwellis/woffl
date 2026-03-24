import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from woffl.pvt import BlackOil, FormGas, FormWater, ResMix


def compute_resmix_data(
    prs_ray: np.ndarray | pd.Series, temp: float, wc: float, fgor: float, oil_api: float, pbub: float, gas_sg: float
) -> dict:
    """Compute Reservoir Mixture

    Create a list of mass and volume fractions for Oil, Water and Gas
    in a mixture. Can be used to compare to results obtained with hysys
    """

    py_oil = BlackOil(oil_api=oil_api, bubblepoint=pbub, gas_sg=gas_sg)
    py_wat = FormWater(wat_sg=1)
    py_gas = FormGas(gas_sg=gas_sg)
    py_mix = ResMix(wc=wc, fgor=fgor, oil=py_oil, wat=py_wat, gas=py_gas)

    mfac_oil, mfac_wat, mfac_gas = [], [], []
    vfac_oil, vfac_wat, vfac_gas = [], [], []
    rho_mix = []

    for prs in prs_ray:
        py_mix = py_mix.condition(prs, temp)

        mfac = py_mix.mass_fract()
        vfac = py_mix.volm_fract()

        mfac_oil.append(mfac[0])
        mfac_wat.append(mfac[1])
        mfac_gas.append(mfac[2])

        vfac_oil.append(vfac[0])
        vfac_wat.append(vfac[1])
        vfac_gas.append(vfac[2])

        rho_mix.append(py_mix.rho_mix())

    pymix = {
        "mass_fracs": {"oil": mfac_oil, "wat": mfac_wat, "gas": mfac_gas},
        "volm_fracs": {"oil": vfac_oil, "wat": vfac_wat, "gas": vfac_gas},
        "rho_mix": rho_mix,
    }
    return pymix


@pytest.fixture(scope="module")
def hysys_resmix():
    hysys_path = Path(__file__).parents[1] / "data" / "hysys_resmix_peng_rob.json"
    with open(hysys_path) as json_file:
        return json.load(json_file)


@pytest.fixture(scope="module")
def python_resmix(hysys_resmix):
    return compute_resmix_data(
        hysys_resmix["pres_psig"],
        hysys_resmix["temp_degf"],
        hysys_resmix["watercut"],
        hysys_resmix["fgor"],
        hysys_resmix["oil_api"],
        hysys_resmix["pbub"],
        hysys_resmix["gas_sg"],
    )


def test_mass_fractions(hysys_resmix, python_resmix) -> None:
    name_frac = "mass_fracs"
    np.testing.assert_allclose(hysys_resmix[name_frac]["oil"], python_resmix[name_frac]["oil"], rtol=0.01)
    np.testing.assert_allclose(hysys_resmix[name_frac]["wat"], python_resmix[name_frac]["wat"], rtol=0.01)
    np.testing.assert_allclose(hysys_resmix[name_frac]["gas"], python_resmix[name_frac]["gas"], rtol=0.06)


def test_volm_fractions(hysys_resmix, python_resmix) -> None:
    name_frac = "volm_fracs"
    np.testing.assert_allclose(hysys_resmix[name_frac]["oil"], python_resmix[name_frac]["oil"], rtol=0.03)
    np.testing.assert_allclose(hysys_resmix[name_frac]["wat"], python_resmix[name_frac]["wat"], rtol=0.04)
    np.testing.assert_allclose(hysys_resmix[name_frac]["gas"], python_resmix[name_frac]["gas"], rtol=0.06)


def test_mixture_density(hysys_resmix, python_resmix) -> None:
    np.testing.assert_allclose(hysys_resmix["rho_mix"], python_resmix["rho_mix"], rtol=0.04)
