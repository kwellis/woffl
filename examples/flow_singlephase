import matplotlib.pyplot as plt
import numpy as np

import woffl.flow.outflow as of
import woffl.flow.singlephase as sp
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# only works if the command python -m examples.outflow is used
# mirror the hysys stuff
md_list = np.linspace(0, 7000, 100)
vd_list = np.linspace(0, 6500, 100)

# well with 600 fgor, 90% wc, 100 bopd
mpu_wat = FormWater(wat_sg=1.0)
print(f"Formation Water Viscosity: {mpu_wat.viscosity():.1f} cP")
qwat_bpd = 5000  # bwpd

wellprof = WellProfile(md_list, vd_list, 7000)
tubing = Pipe(out_dia=4.5, thick=0.5)
casing = Pipe(out_dia=7, thick=0.562, abs_ruff=0.0095)
wellbore = PipeInPipe(tubing, casing)

ptop = 350  # psig
ttop = 100  # deg f

dp_stat = sp.diff_press_static(mpu_wat.density, -1 * wellprof.jetpump_vd)  # power fluid goes down
dp_fric = of.powerfluid_top_down_friction(ptop, ttop, qwat_bpd, mpu_wat, wellbore, wellprof, flowpath="annulus")

print(f"Static dP: {dp_stat:.1f} psi, Friction dP: {dp_fric:.1f} psi, Combo dP: {dp_stat + dp_fric:.1f} psi")
# should be around 2800 PSI static pressure drop and 250 psi frictional pressure drop
