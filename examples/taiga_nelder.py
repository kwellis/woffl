import numpy as np
import pandas as pd

from woffl.assembly import BatchPump  # woffl 2.0
from woffl.flow.inflow import InFlow
from woffl.geometry import JetPump, Pipe, PipeInPipe, WellProfile
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# run from the command line with python -m taiga.oscar202_obd
tubing = Pipe(out_dia=4.5, thick=0.271)
casing = Pipe(out_dia=7.0, thick=0.362)
wbore = PipeInPipe(tubing, casing)

# theoretical well test at 600 PSIG bottom hole?
qoil_std = 2355
bhp = 600
form_wc = 0.08
ipr = InFlow(qwf=qoil_std, pwf=bhp, pres=1900)

mpu_oil = BlackOil(oil_api=21, bubblepoint=1750, gas_sg=0.65)  # 21 API, need to udpate
mpu_wat = FormWater.schrader()
mpu_gas = FormGas.schrader()

wprofile = WellProfile.schrader()
# wprofile.plot_raw()

welhed_pres = 400  # psi, production surf
suct_temp = 75  # deg F
pwrfld_pres = 3200  # psi, power fluid surf

seed_jpump = JetPump("14", "B")
lift_cost = 0.05

results = []
end_gor = 2600
gor_ray = np.linspace(200, end_gor, 5)
for form_gor in gor_ray:
    print(f"Running {form_gor} GOR Case")
    res_mix = ResMix(wc=form_wc, fgor=float(form_gor), oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
    o202_batch = BatchPump(
        pwh=welhed_pres,
        tsu=suct_temp,
        ppf_surf=pwrfld_pres,
        wellbore=wbore,
        wellprof=wprofile,
        ipr_su=ipr,
        prop_su=res_mix,
        prop_pf=mpu_wat,
        jpump_direction="reverse",
        wellname="O-202",
    )
    df = o202_batch.search_run(seed_jpump, lift_cost)
    print(f"{df}\n")
    res_dict = {
        "gor": form_gor,
        "jetpump": df["nozzle"].iloc[0] + df["throat"].iloc[0],
        "bhp_psig": df["psu_solv"].iloc[0],
        "qoil_std": df["qoil_std"].iloc[0],
        "lift_wat": df["lift_wat"].iloc[0],
    }
    results.append(res_dict)

    # update this that if the combo doesn't exist, it doesn't attach it
    # no error gets thrown
    nozs = ["10", "11", "12", "13", "14", "15", "16", "17"]
    thrs = ["A", "B", "C", "D", "E"]

    jp_list = BatchPump.jetpump_list(nozs, thrs)

    df = o202_batch.batch_run(jp_list, debug=False)
    df = o202_batch.process_results()

    o202_batch.plot_data(water="lift", curve=True)

"""df_gor = pd.DataFrame(results)
print(df_gor)

nozs = ["10", "11", "12", "13", "14", "15", "16", "17"]
thrs = ["A", "B", "C", "D"]

jp_list = BatchPump.jetpump_list(nozs, thrs)

df = o202_batch.batch_run(jp_list, debug=False)
df = o202_batch.process_results()

o202_batch.plot_data(water="lift", curve=True)"""
