import matplotlib.pyplot as plt
import numpy as np

import woffl.flow.outflow as of
from woffl.geometry.pipe import Pipe, PipeInPipe
from woffl.geometry.wellprofile import WellProfile
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# only works if the command python -m examples.outflow is used
# mirror the hysys stuff
md_list = np.linspace(0, 6000, 100)
vd_list = np.linspace(0, 4000, 100)

# well with 600 fgor, 90% wc, 100 bopd
mpu_oil = BlackOil.schrader()
mpu_wat = FormWater.schrader()
mpu_gas = FormGas.schrader()
form_gor = 600  # scf/stb
form_wc = 0.9
qoil_std = 100  # stbopd

test_prop = ResMix(form_wc, form_gor, mpu_oil, mpu_wat, mpu_gas)
wellprof = WellProfile(md_list, vd_list, 6000)
tubing = Pipe(out_dia=4.5, thick=0.5)
casing = Pipe.seven_case()
wellbore = PipeInPipe(tubing, casing)

ptop = 350  # psig
ttop = 100  # deg f

md_seg, prs_ray, slh_ray = of.production_top_down_press(ptop, ttop, qoil_std, test_prop, wellbore, wellprof)

slh_ray = np.append(slh_ray, np.nan)  # add a nan to make same length for graphing
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(md_seg, prs_ray, linestyle="--", color="b", label="Pressure")
ax1.set_ylabel("Pressure, PSIG")
ax1.set_xlabel("Measured Depth, Feet")
ax2.plot(md_seg, slh_ray, linestyle="-", color="r", label="Holdup")
ax2.set_ylabel("Slip Liquid Holdup")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc=0)
plt.show()

print(f"Bottom Pressure: {round(prs_ray[-1], 2)} psi")
