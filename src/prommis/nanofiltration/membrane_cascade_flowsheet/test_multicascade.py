"""Multiple Cascade Case Study."""
import sys

from pyomo.environ import (
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    value,
)
from pyomo.network import Arc

from idaes.core.util import to_json, from_json
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import report_statistics

from prommis.nanofiltration.membrane_cascade_flowsheet import utils
from prommis.nanofiltration.membrane_cascade_flowsheet.diafiltration_flowsheet_model import (
    DiafiltrationModel,
)
from prommis.nanofiltration.membrane_cascade_flowsheet.multistart_solve import MultistartSolve
from matplotlib import pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition, SolverResults
from idaes.core.util.model_statistics import report_statistics
import pyomo.contrib.pyros as pyros
from pyros_setting import pyros_settings
import logging
import numpy as np
# from small_tilt_ell_unc_setup import unc_setup, construct_cov_mat
import confidence_ellipsoid.confidence_ellipsoid as ce
from multistart_solve import MultistartSolve

import logging

import pyomo.environ as pyo
import pyomo.contrib.pyros as pyros
import numpy as np

from pyomo.contrib.fbbt.fbbt import fbbt

#################################
# deterministic model
#################################

mix_style = "stage"
mix = mix_style

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


#################################
# multiperiod
#################################

# T = 1
# m = df.build_full_flowsheet(mix_style, LiLB=0.7, CoLB=0.7, periods=T)
# # m.cost_objective.deactivate()
# # m.purity_obj = pyo.Objective(expr=m.period[1].purity_li, sense=pyo.maximize)
# # m.period[1].purity_li_lb.activate()
# # m.period[1].pure = 0.65
# solver = SolverFactory("gams:conopt")
# result = solver.solve(m, tee=True)

# print(result)
# for t in pyo.RangeSet(T):
#     # utils.report_values(m.period[t])
#     print(f'Co/Li recoveries for period {t}')
#     print(pyo.value(m.period[t].prec_perc_co))
#     print(pyo.value(m.period[t].prec_perc_li))
# report_statistics(m)
# vals = utils.report_values(m.period[1])
# utils.visualize_flows(
#     num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals
# )
# report_statistics(m)



#################################
# multicascade 
#################################

# set up each cascade block
# cascade1 focus on Li recovery
cascade1 = df.build_cascade_block(mix_style, LiLB=0.7, CoLB=0.5, periods=1)

# cascade2 should be initialized with retentate outlet of cascade1
cascade2_feed = {
    'solvent': cascade1.period[1].fs.precipitator['retentate'].downstream.flow_vol[0].value,
    'Li': cascade1.period[1].fs.precipitator['retentate'].downstream.flow_mass_solute[0, 'Li'].value,
    'Co': cascade1.period[1].fs.precipitator['retentate'].downstream.flow_mass_solute[0, 'Co'].value
}
df.feed = cascade2_feed
cascade2 = df.build_cascade_block(mix_style, LiLB=0.5, CoLB=0.7, periods=1)

# add each cascade to the same model
test_multicascade_m = pyo.ConcreteModel()
test_multicascade_m.c1 = cascade1
test_multicascade_m.c2 = cascade2

# connect cascade1 to cascade2
test_multicascade_m.cascade1_to_cascade2 = Arc(
    source=test_multicascade_m.c1.period[1].fs.precipitator['retentate'].downstream,
    destination=test_multicascade_m.c2.period[1].fs.split_feed.inlet,
)
test_multicascade_m.c2.period[1].fs.split_feed.inlet.unfix()
TransformationFactory("network.expand_arcs").apply_to(test_multicascade_m)

# new objective
test_multicascade_m.overall_cost_objective = pyo.Objective(
    expr=cascade1.cost_objective.expr + cascade2.cost_objective.expr
)
m = test_multicascade_m

solver = SolverFactory("gams:conopt")
result = solver.solve(m, tee=True)

print(result)
report_statistics(m)
vals = utils.report_values(m.c1.period[1])
utils.visualize_flows(
    num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals
)
vals = utils.report_values(m.c2.period[1])
utils.visualize_flows(
    num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals
)

# TODO
# 1. Show initialization design (Co/Li)
# 2. Show optimized design (Co/Li)
# 3. Visualize flowsheet of combined cascades
# 4. Add in Fe for Fe/Co/Li separation (repeat analysis)
# 5. Consider case study with Li brine or semiconductor waste feedstock...
