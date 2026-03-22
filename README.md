![woffl_github7](https://github.com/kwellis/woffl/assets/62774251/8b80146f-a503-4576-8f43-f1aa45d93a05)

Woffl /ˈwɑː.fəl/ is a Python library for numerical modeling and optimization of subsurface jet pump oil wells.

## Installation   

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install woffl.   

```bash
pip install woffl
```   
## Usage   
Defining an oil well in woffl is broken up into different classes that are combined together in an assembly that creates the model. The classes are organized into PVT, Geometry, Flow and Assembly.   

### PVT - Fluid Properties   
The PVT module is used to define the reservoir mixture properties. The classes are BlackOil, FormGas, FormWater and ResMix. BlackOil, FormGas and FormWat are the individual components in a reservoir stream and are fed into a ResMix where the formation gas oil ratio (FGOR) and watercut (WC) are defined.   

```python
from woffl.pvt import BlackOil, FormGas, FormWater, ResMix

foil = BlackOil(oil_api=22, bubblepoint=1750, gas_sg=0.55)
fwat = FormWater(wat_sg=1)
fgas = FormGas(gas_sg=0.55)
fmix = ResMix(wc=0.355, fgor=800, oil=foil, wat=fwat, gas=fgas)
```
A condition of pressure and temperature can be set on individual components or on the ResMix which cascades it to the different components. Different properties can then be calculated. For example with ResMix the streams mass fractions, volumetric fractions, mixture density, component viscosities and mixture speed of sound can be estimated.   

```python
fmix = fmix.condition(press=1500, temp=80)
xoil, xwat, xgas = fmix.mass_fract()
yoil, ywat, ygas = fmix.volm_fract()
dens_mix = fmix.rho_mix()
uoil, uwat, ugas = fmix.visc_comp()
snd_mix = fmix.cmix()
```
If the reader wants to calculate the insitu volumetric flowrates, an oil rate needs to be passed after a condition. The method will calculate the insitu volumetric flowrate for the different components in cubic feet per second. For this method to be accurate, the watercut fraction defined should be to at least three decimal points. EG: 0.355 for 35.5%.    

```python
qoil, qwat, qgas = fmix.insitu_volm_flow(qoil_std=100)
```
### Inflow Performance Relationship (IPR)   

The inflow class is used to define the IPR of the oil well. Either a Vogel or straight line productivity index can be used for predicting the oil rate at a specific wellbore pressure. The inflow class is defined using a known oil rate, flowing bottom hole pressure and reservoir pressure. Oil rate is used instead of a liquid rate. The predicted oil rate can be used in conjuction with a ResMix to predict the flowing water and gas rates.   

```python
from woffl.flow import InFlow

ipr = InFlow(qwf=246, pwf=1049, pres=1400)
qoil_std = ipr.oil_flow(pnew=800, method="vogel")
```
### WellProfile   

The WellProfile class defines the subsurface geometry of the drillout of the well. To define a WellProfile requires a survey of the measured depth, a survey of the vertical depth, and the jetpump measured depth. The WellProfile will then calculate the horizontal step out of the well as well as filtering the profile into a simplified profile.   
```python
from woffl.geometry import WellProfile

md_examp = [0, 50, 150,...]
vd_examp = [0, 49.99, 149.99,...]
wprof = WellProfile(md_list=md_examp, vd_list=vd_examp, jetpump_md=6693)
```
Basic operations can be conducted on the wellprofile, such as interpolating using the measured depth to return a vertical depth or horizontal stepout.   

```python
vd_dpth = wprof.vd_interp(md_dpth=2234)
hd_dist = wprof.hd_interp(md_dpth=2234)
```
The other benefit of the wellprofile is the ability to visual what the wellprofile looks like under the ground. Either the raw data or the filtered data can be plotted for visualization. The commands to use are below.   

```python
wprof.plot_raw()
wprof.plot_filter()
```
### JetPump

The JetPump class defines the geometry of the jet pump. Currently only Champion X (National) pump geometries are defined. The pump is defined by passing a nozzle number and area ratio. Friction loss coefficients for the nozzle, entrance, throat and diffuser are optional arguments.

```python
from woffl.geometry import JetPump

jpump = JetPump(nozzle_no="12", area_ratio="B")
```
### Pipe and PipeInPipe

The Pipe and PipeInPipe classes define the tubing and casing geometry in the well. Two Pipe objects are combined into a PipeInPipe to represent the wellbore, which is used by the solver to account for friction losses in both the tubing and annulus depending on circulation direction.

```python
from woffl.geometry import Pipe, PipeInPipe

tubing = Pipe(out_dia=4.5, thick=0.5)
casing = Pipe(out_dia=6.875, thick=0.5)
wbore = PipeInPipe(inn_pipe=tubing, out_pipe=casing)
```
Simple geometries of the Pipe and PipeInPipe can be accessed, such as the hydraulic diameter and cross sectional area.

```python
tube_id = tubing.inn_dia
tube_area = tubing.inn_area

ann_dhyd = wbore.ann_hyd_dia
ann_area = wbore.ann_area
```

### Assembly - Batch Run

The assembly module combines the previously defined classes into a system that can be solved. The BatchPump class iterates across a grid of nozzle and throat combinations. After running, `process_results()` identifies semi-finalist pumps (Pareto frontier — no other pump makes more oil for less water) and calculates marginal gradients.

```python
from woffl.assembly import BatchPump

nozs = ["9", "10", "11", "12", "13", "14", "15", "16"]
thrs = ["X", "A", "B", "C", "D", "E"]
jp_list = BatchPump.jetpump_list(nozs, thrs)

well = BatchPump(
    pwh=210, tsu=80, ppf_surf=3168,
    wellbore=wbore, wellprof=wprof, ipr_su=ipr, prop_su=fmix,
    prop_pf=fwat, jpump_direction="reverse", wellname="MPE-41",
)

df = well.batch_run(jp_list)
df = well.process_results()
print(df[df["semi"]])

well.plot_data(water="lift", curve=True)
well.plot_derv(water="lift")
```

### Assembly - Search Run

For single-well optimization, `search_run()` uses Nelder-Mead to find the optimal continuous nozzle and throat diameters, then snaps the result to the nearest catalog pump. The `lift_cost` parameter penalizes power fluid usage — 0.0 maximizes oil regardless of water, higher values favor smaller pumps.

```python
seed_jp = JetPump("12", "B")

df = well.search_run(seed_jp, lift_cost=0.03)
print(df[["nozzle", "throat", "qoil_std", "lift_wat"]])
```

### Assembly - Well Network

The WellNetwork class manages multiple wells sharing a common power fluid supply. It uses a multiple-choice knapsack solver (ortools CP-SAT) to select one jet pump per well that maximizes total oil production subject to the shared power fluid capacity.

```python
from woffl.assembly import WellNetwork

# build and solve each well
wells = [well_a, well_b, well_c, well_d]
for w in wells:
    w.batch_run(jp_list)
    w.process_results()

# optimize across the network
network = WellNetwork(pwh_hdr=None, ppf_hdr=None, well_list=wells, pad_name="Echo Pad")
df = network.optimize(qpf_tot=6000)
print(df.to_string(index=False))

# allow shutting in low-value wells
df_si = network.optimize(qpf_tot=6000, allow_shutin=True)
```

## Examples

The `examples/` directory contains runnable scripts demonstrating different workflows:

- `e41_singlepump.py` — Single well, single pump evaluation with detailed jet pump plots
- `e41_batchpump.py` — Grid search over nozzle/throat combos with semi-finalist analysis
- `e41_searchpump.py` — Nelder-Mead optimization with lift cost sensitivity sweep
- `e41_direction.py` — Forward vs reverse circulation comparison
- `echo_optimize.py` — Multi-well network optimization with shared power fluid

## Background

If the reader is interested in the physics and numerical modeling that went into woffl they should read the papers that are listed below. The conference paper and project by Kaelin Ellis provide a discussion on the numerical analysis and history of jet pumps in oil wells. Cunningham set much of the foundational equations that are used in the modeling.

### Relevant Papers   
- Cunningham, R. G., 1974, “Gas Compression With the Liquid Jet Pump,” ASME J Fluids Eng, 96(3), pp. 203–215.
- Cunningham, R. G., 1995, “Liquid Jet Pumps for Two-Phase Flows,” ASME J Fluids Eng, 117(2), pp. 309–316.
- Ellis, K., Awoleke, O., 2025, “Optimizing Power Fluid in Jet Pump Oil Wells,” SPE-224132-MS, April 25, 2025.
- Himr, D., Habán, V., Pochylý, F., 2009, "Sound Speed in the Mixture Water - Air," Engineering Mechanics, Svratka, Czech Republic, May 11–14, 2009, Paper 255, pp. 393-401. 
