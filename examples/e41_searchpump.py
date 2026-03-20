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

pwh = 210  # psi, wellhead pressure
ppf_surf = 3168  # psi, power fluid surf pressure
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
e41_res = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
e41_profile = WellProfile.schrader()

# seed with a mid-range jet pump and let Nelder-Mead find the best continuous size
seed_jp = JetPump("12", "B")

e41_batch = BatchPump(
    pwh, tsu, ppf_surf, wbore, e41_profile, e41_ipr, e41_res, mpu_wat, jpump_direction="reverse", wellname="MPE-41"
)

df = e41_batch.search_run(seed_jp)
print(df)
