#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Executable file for generating and solving diafiltration model."""

import sys

from pyomo.environ import (
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    value,
)

from idaes.core.util import to_json, from_json
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import report_statistics

from prommis.nanofiltration.membrane_cascade_flowsheet import utils
from prommis.nanofiltration.membrane_cascade_flowsheet.diafiltration_flowsheet_model import (
    DiafiltrationModel,
)
from multistart_solve import MultistartSolve

mix_style = "stage"

def solve_scaled_model(m, L, C):
    m.recovery_li = L
    m.recovery_co = C

    scaling = TransformationFactory("core.scale_model")
    solver = SolverFactory("ipopt")
    solver = SolverFactory('multistart_solve')
    solver.mix = mix_style
    solver.dsolver = 'gams:conopt'

    scaled_model = scaling.create_using(m, rename=False)
    result = solver.solve(scaled_model, tee=False)
    assert_optimal_termination(result)
    # Propagate results back to unscaled model
    scaling.propagate_solution(scaled_model, m)

    return result


"""Driver for creating diafiltration model."""
# collect arguments
# check if arguments are given. Use default if not
num_s = 3
num_t = 10

# set relevant parameter values
solutes = ["Li", "Co"]
flux = 0.1  # m3 / m2 / h
sieving_coefficient = {"Li": 1.3, "Co": 0.5}
feed = {
    "solvent": 100,  # m^3/hr of water
    "Li": 1.7 * 100,  # kg/hr
    "Co": 17 * 100,  # kg/hr
}
diaf = {
    "solvent": 30,  # m^3/hr of water
    "Li": 0.1 * 30,  # kg/hr
    "Co": 0.2 * 30,  # kg/hr
}
precipitate = True

# setup for diafiltration model
df = DiafiltrationModel(
    NS=num_s,
    NT=num_t,
    solutes=solutes,
    flux=flux,
    sieving_coefficient=sieving_coefficient,
    feed=feed,
    diafiltrate=diaf,
    precipitate=precipitate,
    precipitate_yield={
        "permeate": {"Li": 0.81, "Co": 0.01},
        "retentate": {"Li": 0.20, "Co": 0.89},
    },
)

# model initialization
m = df.build_flowsheet(mixing=mix_style)

saved_initialization = False
if saved_initialization:
    from_json(m, fname="initialized_model_stage_3_10")
else:
    df.initialize(m, mixing=mix_style, precipitate=precipitate)
    to_json(m, fname="initialized_model_stage_3_10")

df.unfix_dof(m, mixing=mix_style, precipitate=precipitate)
m.fs.split_diafiltrate.inlet.flow_vol.setub(2000)
# m.fs.precipitator['retentate'].split_inlet['bypass'].fix(0)
# m.fs.precipitator['permeate'].split_inlet['bypass'].fix(0)
report_statistics(m)

costing = True
atmospheric_pressure = 101.325  # ambient pressure, kPa
operating_pressure = 145  # nanofiltration operating pressure, psi
simple_costing = False
npv = False
if costing:
    df.add_costing(
        m,
        NS=num_s,
        flux=flux,
        feed=feed,
        diaf=diaf,
        precipitate=precipitate,
        atmospheric_pressure=atmospheric_pressure,
        operating_pressure=operating_pressure,
        simple_costing=simple_costing,
        npv=npv,
    )
    df.add_costing_objectives(m, npv=npv)
    # df.add_costing_scaling(m, NS=num_s, simple_costing=simple_costing)

report_statistics(m)

# set recovery lower bounds
lithium_recovery = 0.7
cobalt_recovery = 0.7
m.fs.lithium_carbonate_price = 0
m.fs.cobalt_oxalate_price = 0
 
solve_scaled_model(
    m,
    L=lithium_recovery,
    C=cobalt_recovery,
)

# dt = DiagnosticsToolbox(m)
# some flows are at their bounds of zero
# dt.report_numerical_issues()

if costing:
    if not simple_costing:
        # Verify the feed pump operating pressure workaround is valid
        # assume this additional cost is less than half a cent
        if value(m.fs.feed_pump.costing.variable_operating_cost) >= 0.005:
            raise ValueError(
                "The variable  operating cost of the feed pump as calculated in the feed"
                "pump costing block is not negligible. This operating cost is already"
                "accounted for via the membrane's pressure drop specific energy consumption."
            )

# NOTE These percent recoveries are for precipitators
m.prec_perc_co.display()
m.prec_perc_li.display()

# m.fs.costing.total_annualized_cost.display()

# Print all relevant flow information
vals = utils.report_values(m)
utils.visualize_flows(
    num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals
)

import pyomo.environ as pyo

T = 15
avg_li = 0.7
avg_co = 0.7

m = df.build_full_flowsheet(mix_style, avg_li, avg_co, T)
# m_init = df.build_full_flowsheet(mix_style, avg_li, avg_co, 1)
# solver = SolverFactory('ipopt')
# result = solver.solve(m_init, tee=True)
# for t in range(1,T):
#     m = df.build_full_flowsheet(mix_style, avg_li, avg_co, t+1)
#     report_statistics(m)
#     # copy solution of previous model
#     for var in m_init.component_data_objects(pyo.Var):
#         new_var = m.find_component(var.name)
#         if new_var is not None:
#             new_var.value = var.value
#     solver = SolverFactory('ipopt')
#     result = solver.solve(m, tee=True)

#     # set current solution as initialization for next model
#     m_init = m
    
# m = df.build_full_flowsheet(mix_style, 0.5, 0., 15)
import numpy as np
# co_price = np.linspace(40,0, 15)
for t in m.period:
    # m.period[t].cobalt_oxalate_price.pprint()
    # print(co_price[t-1])
    # m.period[t].fs.cobalt_oxalate_price= float(co_price[t-1])
    m.period[t].fs.cobalt_oxalate_price= 7
    m.period[t].fs.lithium_carbonate_price= 1
    m.period[t].fs.ammonium_oxalate_price= 3.1
    m.period[t].fs.soda_ash_price= 0.13
    # m.period[t].fs.ammonium_oxalate_price= 3.1 + 23
    # m.period[t].fs.soda_ash_price= 0.13 + 7
    m.period[t].fs.split_feed.mixed_state[0].flow_vol.fix(100)
    m.period[t].fs.split_feed.mixed_state[0].flow_mass_solute['Co'].fix(1700/2)
    m.period[t].fs.split_feed.mixed_state[0].flow_mass_solute['Li'].fix(170/2)
    # m.period[t].fs.precipitator['retentate'].split_inlet['bypass'].fix(0)
    # m.period[t].fs.precipitator['permeate'].split_inlet['bypass'].fix(0)
    # m.period[t].cobalt_oxalate_price.pprint()

report_statistics(m)
solver = SolverFactory('ipopt')
result = solver.solve(m, tee=True)

print(pyo.value(m.Li_recovery))
print(pyo.value(m.Co_recovery))

# plotting results
import numpy as np
import matplotlib.pyplot as plt

years = np.arange(1, T+1)
plt.figure(figsize=(10,6))
rec_li = np.array([pyo.value(m.period[t].prec_perc_li)*100 for t in m.period])
rec_co = np.array([pyo.value(m.period[t].prec_perc_co)*100 for t in m.period])
plt.plot(years, rec_co)
plt.plot(years, rec_li)
plt.legend(['Co Recovery', 'Li Recovery'])
plt.xlabel("Year")
plt.ylabel("Recovery [%]")
plt.tight_layout()
plt.show()


# Years 0 ... T
years = np.arange(0, T+1)

# Cash flow components (already calculated values)
capital_costs   = np.array([pyo.value(m.period[1].fs.costing.pv_capital_cost)] + [0]*T)
operating_costs = np.array([0] + [pyo.value(m.period[t].fs.costing.pv_operating_cost) for t in m.period])
loan_costs      = np.array([0] + [pyo.value(m.period[t].fs.costing.pv_loan_interest) for t in m.period])
revenue         = np.array([0] + [pyo.value(m.period[t].fs.costing.pv_revenue) for t in m.period])

x = np.arange(len(years))
width = 0.2

plt.figure(figsize=(10,6))

plt.bar(x - 1.5*width, capital_costs, width, label="Capital Costs")
plt.bar(x - 0.5*width, operating_costs, width, label="Operating Costs")
plt.bar(x + 0.5*width, loan_costs, width, label="Loan Costs")
plt.bar(x + 1.5*width, revenue, width, label="Revenue")

plt.axhline(0, color='black', linewidth=0.8)
plt.xticks(x, years)
plt.xlabel("Year")
plt.ylabel("Cash Flow [$MM]")
# plt.title("Cash Flow Components by Year")
plt.legend()

# Display NPV on the plot
plt.text(
    0.4, 0.95,
    f"NPV = {pyo.value(m.cost_objective):,.0f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment="top",
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
)

plt.tight_layout()
plt.show()
