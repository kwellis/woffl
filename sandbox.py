"""from flow import jetflow as jf
from flow import jetplot as jplt
from flow import singlephase as sp
from flow.inflow import InFlow
from geometry.jetpump import JetPump
from geometry.pipe import Annulus, Pipe
from geometry.wellprofile import WellProfile
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

# only works if the command python -m tests.wprof_test is used

# imported values from e42
md_list = [
    0.0,
    100.0,
    165.0,
    228.0,
    291.0,
    353.0,
    416.0,
    478.0,
    540.0,
    603.0,
    668.47,
    729.76,
    792.38,
    857.25,
    920.91,
    984.6,
    1049.15,
    1113.1,
    1177.05,
    1240.07,
    1303.41,
    1367.02,
    1430.27,
    1494.48,
    1557.55,
    1621.14,
    1684.77,
    1749.15,
    1812.58,
    1874.88,
    1939.92,
    2004.37,
    2067.46,
    2130.86,
    2194.94,
    2258.7,
    2320.87,
    2385.63,
    2449.12,
    2513.13,
    2576.48,
    2639.95,
    2703.65,
    2766.87,
    2831.08,
    2894.74,
    2958.25,
    3021.82,
    3085.63,
    3148.98,
    3212.81,
    3276.35,
    3339.94,
    3403.24,
    3467.48,
    3531.39,
    3595.28,
    3659.17,
    3722.71,
    3786.56,
    3850.03,
    3913.72,
    3978.01,
    4041.17,
    4104.31,
    4168.21,
    4231.85,
    4295.51,
    4359.66,
    4423.27,
    4487.4,
    4550.98,
    4614.5,
    4677.75,
    4741.46,
    4805.42,
    4869.03,
    4932.88,
    4995.97,
    5060.18,
    5124.3,
    5187.84,
    5251.05,
    5314.67,
    5378.25,
    5442.06,
    5505.96,
    5569.28,
    5633.03,
    5697.02,
    5760.39,
    5823.75,
    5887.4,
    5951.38,
    6015.04,
    6078.96,
    6142.4,
    6206.58,
    6270.37,
    6333.91,
    6397.75,
    6459.84,
    6524.75,
    6586.71,
    6652.49,
    6715.28,
    6780.34,
    6843.97,
    6907.48,
    6971.45,
    7035.26,
    7099.07,
    7160.77,
    7226.26,
    7290.11,
    7353.89,
    7417.85,
    7481.54,
    7545.28,
    7608.7,
    7672.42,
    7736.11,
    7800.37,
    7863.49,
    7927.25,
    7988.2,
    8054.5,
    8118.07,
    8181.68,
    8245.46,
    8281.05,
    8343.89,
    8407.38,
    8470.54,
    8534.02,
    8597.81,
    8661.58,
    8725.35,
    8789.48,
    8852.84,
    8916.62,
    8980.32,
    9043.87,
    9107.23,
    9171.59,
    9234.96,
    9299.44,
    9362.49,
    9426.1,
    9490.11,
    9553.21,
    9615.38,
    9678.84,
    9742.64,
    9808.66,
    9872.23,
    9935.41,
    9999.73,
    10063.41,
    10126.86,
    10188.82,
    10253.69,
    10316.73,
    10382.01,
    10445.74,
    10509.19,
    10573.15,
    10635.73,
    10700.83,
    10764.12,
    10828.26,
    10891.55,
    10955.51,
    11018.96,
    11082.5,
    11146.2,
    11210.22,
    11273.59,
    11337.62,
    11400.98,
    11465.32,
    11528.78,
    11592.23,
    11655.78,
    11719.52,
    11783.84,
    11847.09,
    11899.06,
    11969.0,
]

tvd_list = [
    0.0,
    99.99993,
    164.9998,
    227.99935,
    290.99826,
    352.98873,
    415.90045,
    477.62711,
    539.09468,
    601.28561,
    665.62471,
    725.64553,
    786.6839,
    849.23387,
    909.72895,
    969.52142,
    1029.12834,
    1086.886,
    1142.92035,
    1195.86979,
    1246.65495,
    1294.72035,
    1340.50473,
    1385.43381,
    1428.37523,
    1471.40974,
    1513.13005,
    1553.35537,
    1591.27876,
    1626.59965,
    1661.85597,
    1696.10009,
    1730.24438,
    1765.41259,
    1800.31269,
    1834.35507,
    1867.03756,
    1901.25381,
    1934.75651,
    1967.46557,
    1998.69814,
    2029.40142,
    2060.78866,
    2093.37566,
    2126.38611,
    2157.8066,
    2188.78748,
    2219.87351,
    2251.48421,
    2283.33145,
    2316.59177,
    2349.52493,
    2381.89976,
    2414.34083,
    2446.81977,
    2478.95863,
    2511.55347,
    2545.04961,
    2578.31487,
    2611.12872,
    2643.90437,
    2676.51855,
    2708.89199,
    2740.8343,
    2773.53295,
    2806.82566,
    2839.99234,
    2873.28335,
    2906.88292,
    2939.94386,
    2972.93521,
    3005.71975,
    3039.41617,
    3073.06795,
    3105.89559,
    3138.52615,
    3171.34953,
    3204.22531,
    3236.33149,
    3269.27263,
    3302.01382,
    3334.32526,
    3365.67686,
    3396.43824,
    3427.01004,
    3457.3062,
    3487.8702,
    3519.25651,
    3552.14241,
    3585.66789,
    3618.43855,
    3651.30836,
    3684.69826,
    3718.18518,
    3751.55187,
    3785.14034,
    3818.61786,
    3852.53822,
    3886.05832,
    3919.55078,
    3952.98397,
    3984.90721,
    4017.83255,
    4047.81272,
    4078.69479,
    4106.71301,
    4132.14625,
    4153.88145,
    4172.14988,
    4185.70322,
    4195.60371,
    4203.70759,
    4210.84246,
    4218.72777,
    4226.75847,
    4234.71963,
    4242.5372,
    4250.67454,
    4258.59207,
    4265.81047,
    4273.01874,
    4280.80869,
    4290.62797,
    4302.23394,
    4313.94673,
    4324.38428,
    4334.55034,
    4342.87575,
    4348.3251,
    4351.59135,
    4353.64033,
    4357.31289,
    4360.21012,
    4361.49475,
    4360.70317,
    4358.36067,
    4355.36835,
    4352.37595,
    4349.40025,
    4345.94678,
    4341.85368,
    4337.35518,
    4333.00541,
    4329.04963,
    4325.62009,
    4322.51388,
    4319.17913,
    4315.88538,
    4311.25489,
    4305.55973,
    4300.31863,
    4295.45757,
    4290.87671,
    4286.85457,
    4283.08346,
    4279.96756,
    4277.0745,
    4274.19653,
    4271.69128,
    4269.77629,
    4268.20353,
    4266.69839,
    4265.88465,
    4264.86553,
    4263.4202,
    4261.29495,
    4258.29371,
    4255.05179,
    4251.50345,
    4247.81667,
    4243.87361,
    4240.77147,
    4239.12004,
    4238.77728,
    4238.91088,
    4238.01088,
    4236.00046,
    4233.81152,
    4231.047,
    4227.86384,
    4224.07663,
    4219.49024,
    4214.94324,
    4210.66554,
    4206.88594,
    4203.83414,
    4200.62917,
    4197.74213,
    4193.97265,
]

e42_profile = WellProfile(md_list, tvd_list, 7000)
print(e42_profile.outflow_spacing(100))
# e42_profile.plot_raw()
e42_profile.plot_filter()
"""

import matplotlib.pyplot as plt
import numpy as np

import flow.outflow as of
from geometry.pipe import Pipe
from geometry.wellprofile import WellProfile
from pvt.blackoil import BlackOil
from pvt.formgas import FormGas
from pvt.formwat import FormWater
from pvt.resmix import ResMix

# only works if the command python -m tests.outflow_test is used
# mirror the hysys stuff
md_list = np.linspace(0, 6000, 100)
vd_list = np.linspace(0, 4000, 100)

# well with 600 fgor, 90% wc, 200 bopd
mpu_oil = BlackOil.schrader_oil()
mpu_wat = FormWater.schrader_wat()
mpu_gas = FormGas.schrader_gas()
form_gor = 600  # scf/stb
form_wc = 0.9
qoil_std = 100  # stbopd

test_prop = ResMix(form_wc, form_gor, mpu_oil, mpu_wat, mpu_gas)
wellprof = WellProfile(md_list, vd_list, 6000)
tubing = Pipe(out_dia=4.5, thick=0.237)

ptop = 400  # psig
ttop = 100  # deg f

md_seg, prs_ray, slh_ray = of.top_down_press(ptop, ttop, qoil_std, test_prop, tubing, wellprof)

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
