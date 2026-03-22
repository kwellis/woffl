"""MCKP Example: Multi-well jet pump optimization

Demonstrates the multiple-choice knapsack solver for allocating power fluid
across a network of wells. Each well runs a batch of jet pumps, then MCKP
picks one pump per well to maximize total oil subject to shared capacity.
"""

from woffl.assembly import BatchPump, WellNetwork
from woffl.flow.inflow import InFlow
from woffl.geometry import JetPump, Pipe, PipeInPipe, WellProfile
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

# shared infrastructure
pwh = 210  # psi, wellhead pressure
ppf_surf = 3168  # psi, power fluid surface pressure
tsu = 80  # F, suction temperature

tubing = Pipe(out_dia=4.5, thick=0.5)
casing = Pipe(out_dia=6.875, thick=0.5)
wbore = PipeInPipe(inn_pipe=tubing, out_pipe=casing)
profile = WellProfile.schrader()

mpu_oil = BlackOil.schrader()
mpu_wat = FormWater.schrader()
mpu_gas = FormGas.schrader()

# well definitions — same pad, different IPRs and water cuts
well_configs = [
    {"name": "MPE-41", "qwf": 246, "pwf": 1049, "pres": 1400, "wc": 0.894, "gor": 600},
    {"name": "MPE-42", "qwf": 180, "pwf": 1100, "pres": 1350, "wc": 0.920, "gor": 550},
    {"name": "MPE-43", "qwf": 310, "pwf": 950, "pres": 1450, "wc": 0.850, "gor": 650},
    {"name": "MPE-44", "qwf": 120, "pwf": 1150, "pres": 1300, "wc": 0.950, "gor": 500},
]

# jet pumps to evaluate
nozs = ["9", "10", "11", "12", "13", "14"]
thrs = ["A", "B", "C", "D", "E"]
jp_list = BatchPump.jetpump_list(nozs, thrs)

# build wells, run batch, process results
wells = []
for cfg in well_configs:
    ipr = InFlow(qwf=cfg["qwf"], pwf=cfg["pwf"], pres=cfg["pres"])
    res = ResMix(wc=cfg["wc"], fgor=cfg["gor"], oil=mpu_oil, wat=mpu_wat, gas=mpu_gas)
    well = BatchPump(
        pwh,
        tsu,
        ppf_surf,
        wbore,
        profile,
        ipr,
        res,
        mpu_wat,
        jpump_direction="reverse",
        wellname=cfg["name"],
    )
    well.batch_run(jp_list)
    well.process_results()
    wells.append(well)
    semis = well.df["semi"].sum()
    print(f"{cfg['name']}: {semis} semi-finalists from {len(jp_list)} pumps")

# build network and optimize — every well must pump
qpf_tot = 6000  # total available power fluid, bwpd
network = WellNetwork(pwh_hdr=None, ppf_hdr=None, well_list=wells, pad_name="Echo Pad")
df = network.optimize(qpf_tot)

print("\n=== MCKP Solution (All wells online) ===")
print(df.to_string(index=False))
print(f"\nTotal oil:   {df['qoil_std'].sum():.1f} bopd")
print(f"Total water: {df['lift_wat'].sum():.1f} / {qpf_tot:.0f} bwpd")

# optimize — solver can shutin crummy wells
df_si = network.optimize(qpf_tot, allow_shutin=True)

print("\n=== MCKP Solution (Shutin allowed) ===")
print(df_si.to_string(index=False))
print(f"\nTotal oil:   {df_si['qoil_std'].sum():.1f} bopd")
print(f"Total water: {df_si['lift_wat'].sum():.1f} / {qpf_tot:.0f} bwpd")
