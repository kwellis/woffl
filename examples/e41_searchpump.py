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

seed_jp = JetPump("12", "B")

e41_batch = BatchPump(
    pwh, tsu, ppf_surf, wbore, e41_profile, e41_ipr, e41_res, mpu_wat, jpump_direction="reverse", wellname="MPE-41"
)

# first run the full batch to see the semi-finalist molwr values
nozs = ["9", "10", "11", "12", "13", "14", "15", "16"]
thrs = ["X", "A", "B", "C", "D", "E"]
jp_list = BatchPump.jetpump_list(nozs, thrs)
df_batch = e41_batch.batch_run(jp_list)
df_batch = e41_batch.process_results()

semi = df_batch[df_batch["semi"]][["nozzle", "throat", "qoil_std", "lift_wat", "molwr"]]
print("=== Semi-Finalist MOLWR Values ===")
print(semi.to_string(index=False))
print()

# now run search_run with different lift_cost penalties and see which pump it picks
lift_costs = [0.0, 0.01, 0.02, 0.03, 0.04]

print("=== Search Run with Different MOLWR Costs ===")
for cost in lift_costs:
    df = e41_batch.search_run(seed_jp, lift_cost=cost)
    noz = df["nozzle"].iloc[0]
    thr = df["throat"].iloc[0]
    qoil = df["qoil_std"].iloc[0]
    lwat = df["lift_wat"].iloc[0]
    print(f"lift_cost={cost:.2f}  =>  {noz}{thr}  oil={qoil:.1f}  lift_wat={lwat:.1f}")

e41_batch.plot_data("lift", True)
