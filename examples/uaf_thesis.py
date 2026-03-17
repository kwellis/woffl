import time
import timeit

import numpy as np

from woffl.optimization import network as nw
from woffl.optimization import ratiotest as rt
from woffl.optimization import rednewton as rn

# define the wells and define the minimum power fluid flowrate in BPD
wells = {
    "india_15": {"c1": 353, "c2": 230, "c3": 9.6e-4, "qpf_min": 0, "qpf_max": 8000},
    "india_17": {"c1": 578, "c2": 430, "c3": 5.6e-4, "qpf_min": 0, "qpf_max": 8000},
    "india_33": {"c1": 1237, "c2": 1226, "c3": 7.35e-4, "qpf_min": 0, "qpf_max": 8000},
    "india_31": {"c1": 944, "c2": 807, "c3": 6.9e-4, "qpf_min": 0, "qpf_max": 8000},
}

# need to update this so it doesn't exceed max powerfluid constraints?
# if it does exceed, then this constraint is just inactive for the duration?
# also, it will be a pretty short optimization run? ha
Qp_tot = 12500  # max available water flow in the system

t0 = time.time()
Qo, Qp, dfk, k = nw.optimize_power_fluid(wells, Qp_tot)
t1 = time.time()

dur = timeit.timeit(lambda: nw.optimize_power_fluid(wells, Qp_tot), number=10)

print(f"\nRequired Iterations: {k}")
print(f"Total Oil Rate: {-1 * Qo:.2f}")
print(f"Distributed Power Fluid:\n{Qp}")
print(f"Gradient at each well:\n{dfk}")
print(f"Sum Individual Power Fluid: {sum(Qp):.2f}")
print(f"Took: {dur}")
