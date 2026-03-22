"""Shared fixtures for assembly tests

Common well setup and helpers used across solopump, batchpump, and network tests.
"""

import pytest

from woffl.assembly.batchpump import BatchPump
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# --- shared well setup (MPU E-41 reference) ---

pwh = 210  # psig, wellhead pressure
ppf_surf = 3168  # psig, power fluid surface pressure
tsu = 80  # deg F, suction temperature

tubing = Pipe(out_dia=4.5, thick=0.5)
casing = Pipe(out_dia=6.875, thick=0.5)
wbore = PipeInPipe(inn_pipe=tubing, out_pipe=casing)
profile = WellProfile.schrader()

mpu_oil = BlackOil.schrader()
mpu_wat = FormWater.schrader()
mpu_gas = FormGas.schrader()

# small grid to keep tests fast
nozzles = ["10", "11", "12", "13"]
throats = ["A", "B", "C", "D"]
jp_list = BatchPump.jetpump_list(nozzles, throats)


def make_well(name, qwf, pwf, pres, wc, fgor):
    """Helper to build a BatchPump from well parameters."""
    ipr = InFlow(qwf=qwf, pwf=pwf, pres=pres)
    res = ResMix(wc=wc, fgor=fgor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
    return BatchPump(
        pwh, tsu, ppf_surf, wbore, profile, ipr, res, mpu_wat,
        jpump_direction="reverse", wellname=name,
    )
