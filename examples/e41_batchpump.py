import numpy as np
import pandas as pd

from woffl.assembly.batchrun import BatchPump
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# data from MPU E-41 Well Test on 11/27/2023
# only works if the command python -m examples.batch is used

pwh = 210  # psi, wellhead pressure
jpump_tvd = 4065  # feet, interpolated off well profile
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # psi, power fluid surf pressure 3168
tsu = 80

# testing the jet pump code on E-41
tubing = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
casing = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
wbore = PipeInPipe(inn_pipe=tubing, out_pipe=casing)  # define the wellbore

e41_ipr = InFlow(qwf=246, pwf=1049, pres=1400)  # define an ipr

mpu_oil = BlackOil.schrader()  # class method
mpu_wat = FormWater.schrader()  # class method
mpu_gas = FormGas.schrader()  # class method

form_wc = 0.894
form_gor = 600  # formation gor
form_temp = 111
e41_res = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
e41_profile = WellProfile.schrader()

nozs = ["9", "10", "11", "12", "13", "14", "15", "16"]
thrs = ["X", "A", "B", "C", "D", "E"]

jp_list = BatchPump.jetpump_list(nozs, thrs)
e41_batch = BatchPump(
    pwh, tsu, ppf_surf, wbore, e41_profile, e41_ipr, e41_res, mpu_wat, jpump_direction="reverse", wellname="MPE-41"
)

df = e41_batch.batch_run(jp_list)
df = e41_batch.process_results()
print(df[df["semi"]])

e41_batch.plot_data(water="lift", curve=True)
e41_batch.plot_derv(water="lift")
