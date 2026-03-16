import numpy as np

from woffl.flow import jetflow as jf
from woffl.flow import jetgraphs as jg
from woffl.flow import jetplot as jplt
from woffl.flow.inflow import InFlow
from woffl.geometry.jetpump import JetPump
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt.blackoil import BlackOil
from woffl.pvt.formgas import FormGas
from woffl.pvt.formwat import FormWater
from woffl.pvt.resmix import ResMix

# data from MPU E-41 Well Test on 11/27/2023
# only works if the command python -m examples.e41 is used

surf_pres = 210
jpump_tvd = 4065  # feet, interpolated off well profile
rho_pf = 62.4  # lbm/ft3
ppf_surf = 3168  # psi, power fluid surf pressure 3168

# testing the jet pump code on E-41
tube = Pipe(out_dia=4.5, thick=0.5)  # E-42 tubing
case = Pipe(out_dia=6.875, thick=0.5)  # E-42 casing
ann = PipeInPipe(inn_pipe=tube, out_pipe=case)  # define the annulus

e41_ipr = InFlow(qwf=246, pwf=1049, pres=1400)  # define an ipr

e41_jp = JetPump(nozzle_no="13", area_ratio="A", ken=0.03, kth=0.3, kdi=0.4)

mpu_oil = BlackOil.schrader()  # class method
mpu_wat = FormWater.schrader()  # class method
mpu_gas = FormGas.schrader()  # class method

form_wc = 0.894
form_gor = 600  # formation gor
form_temp = 111
e41_res = ResMix(wc=form_wc, fgor=form_gor, oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)

e42_profile = WellProfile.schrader()

jg.choked_figures(form_temp, rho_pf, ppf_surf, e41_jp, tube, e42_profile, e41_ipr, e41_res)
# jg.discharge_check(surf_pres, form_temp, rho_pf, ppf_surf, e41_jp, tube, e42_profile, e41_ipr, e41_res)

psu_ray = np.linspace(1106, 1250, 5)
qoil_list, book_list = jplt.multi_throat_entry_books(psu_ray, form_temp, e41_jp.ken, e41_jp.ate, e41_ipr, e41_res)
jplt.multi_suction_graphs(qoil_list, book_list)
