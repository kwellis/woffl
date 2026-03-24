import json
from pathlib import Path

import numpy as np
import pytest

from woffl.pvt.formgas import FormGas


def compute_formgas_data(prs_ray: np.ndarray | list, temp: float, gas_sg: float) -> dict:
    """Compute FormGas Data

    Create a list of properties of a formgas. Can be used to compare to results obtained with hysys.
    Density, viscosity and zfactor.
    """
    py_gas = FormGas(gas_sg=gas_sg)
    rho_gas, visc_gas, zfactor = [], [], []

    for prs in prs_ray:
        py_gas = py_gas.condition(prs, temp)

        rho_gas.append(py_gas.density)
        visc_gas.append(py_gas.viscosity)
        zfactor.append(py_gas.zfactor)

    pygas = {"rho_gas": rho_gas, "visc_gas": visc_gas, "zfactor": zfactor}
    return pygas


@pytest.fixture(scope="module")
def hysys_formgas():
    hysys_path = Path(__file__).parents[1] / "data" / "hysys_formgas_peng_rob.json"
    with open(hysys_path) as json_file:
        return json.load(json_file)


@pytest.fixture(scope="module")
def python_formgas(hysys_formgas):
    return compute_formgas_data(hysys_formgas["pres_psig"], hysys_formgas["temp_degf"], hysys_formgas["gas_sg"])


# blasingame 1988 PDF
blasing_ppr = 2.958
blasing_tpr = 1.867
blasing_zf = 0.9117


def test_zfactor_gradschool() -> None:
    grad_zf = FormGas._zfactor_grad_school(blasing_ppr, blasing_tpr)
    assert grad_zf == pytest.approx(blasing_zf, rel=0.01)


def test_zfactor_dranchuk() -> None:
    dak_zf = FormGas._zfactor_dak(blasing_ppr, blasing_tpr)
    assert dak_zf == pytest.approx(blasing_zf, rel=0.01)


def test_gas_compressibility() -> None:
    # properties of petroleum fluids, 2nd Edition, McCain, Pag 174
    pres_mccain = 1000 + 14.7  # psia + atm
    temp_mccain = 68
    mccain_compress = 0.001120
    methane = FormGas(gas_sg=0.55)
    calc_compress = methane.condition(pres_mccain, temp_mccain).compress
    assert calc_compress == pytest.approx(mccain_compress, rel=0.03)


def test_gas_density(hysys_formgas, python_formgas) -> None:
    np.testing.assert_allclose(hysys_formgas["rho_gas"], python_formgas["rho_gas"], rtol=0.05)


def test_gas_viscosity(hysys_formgas, python_formgas) -> None:
    np.testing.assert_allclose(hysys_formgas["visc_gas"], python_formgas["visc_gas"], rtol=0.03)


def test_gas_zfactor(hysys_formgas, python_formgas) -> None:
    np.testing.assert_allclose(hysys_formgas["zfactor"], python_formgas["zfactor"], rtol=0.04)
