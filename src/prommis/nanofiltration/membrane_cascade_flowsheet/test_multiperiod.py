"""Manufacturing Variability Case Study."""
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


# # model initialization
# m = df.build_flowsheet(mixing=mix_style)

# saved_initialization = False
# if saved_initialization:
#     from_json(m, fname="initialized_model_stage_3_10")
# else:
#     df.initialize(m, mixing=mix_style, precipitate=precipitate)
#     to_json(m, fname="initialized_model_stage_3_10")

# df.unfix_dof(m, mixing=mix_style, precipitate=precipitate)
# m.fs.split_diafiltrate.inlet.flow_vol.setub(2000)
# m.fs.split_diafiltrate.inlet.flow_vol.setlb(1e-11)
# report_statistics(m)

# costing = True
# atmospheric_pressure = 101.325  # ambient pressure, kPa
# operating_pressure = 145  # nanofiltration operating pressure, psi
# simple_costing = False
# npv = False
# if costing:
#     df.add_costing(
#         m,
#         NS=num_s,
#         flux=flux,
#         feed=feed,
#         diaf=diaf,
#         precipitate=precipitate,
#         atmospheric_pressure=atmospheric_pressure,
#         operating_pressure=operating_pressure,
#         simple_costing=simple_costing,
#         npv=npv,
#     )
#     df.add_costing_objectives(m, npv=npv)
#     # df.add_costing_scaling(m, NS=num_s, simple_costing=simple_costing)

# # set recovery lower bounds
# m.fs.soda_ash_price = 0
# m.fs.ammonium_oxalate_price = 0
# m.fs.lithium_carbonate_price = 0
# m.fs.cobalt_oxalate_price = 0
# # m.recovery_li = 0
# # m.recovery_co = 0

# # # solve for initial feasible solution
# # solver = SolverFactory("ipopt")
# # result = solver.solve(m, tee=False)

# # # tighten bounds
# # fbbt(m)

# m.recovery_li = 0.7
# m.recovery_co = 0.7

# # solver = SolverFactory("ipopt")
# # solver = SolverFactory("gams:conopt")
# # result = solver.solve(m, tee=False)
# # m.recovery_li = 0.5
# # m.recovery_co = 0.5
# solver = SolverFactory('multistart_solve')
# solver.mix = mix_style
# # solver.max_sense = True
# solver.dsolver = 'gams:conopt'

# result = solver.solve(m, tee=False)
# # result = solver.solve(m, tee=True)
# assert_optimal_termination(result)

# # dt = DiagnosticsToolbox(m)
# # some flows are at their bounds of zero
# # dt.report_numerical_issues()

# if costing:
#     if not simple_costing:
#         # Verify the feed pump operating pressure workaround is valid
#         # assume this additional cost is less than half a cent
#         if value(m.fs.feed_pump.costing.variable_operating_cost) >= 0.005:
#             raise ValueError(
#                 "The variable  operating cost of the feed pump as calculated in the feed"
#                 "pump costing block is not negligible. This operating cost is already"
#                 "accounted for via the membrane's pressure drop specific energy consumption."
#             )

# # NOTE These percent recoveries are for precipitators
# m.prec_perc_co.display()
# m.prec_perc_li.display()

# # m.fs.costing.total_annualized_cost.display()

# # Print all relevant flow information
# vals = utils.report_values(m)
# utils.visualize_flows(
#     num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals
# )




#################################
# multiperiod
#################################

T = 8
m = df.build_full_flowsheet(mix_style, LiLB=0.7, CoLB=0.7, periods=T)
# m.cost_objective.deactivate()
# m.purity_obj = pyo.Objective(expr=m.period[1].purity_li, sense=pyo.maximize)
# m.period[1].purity_li_lb.activate()
# m.period[1].pure = 0.65
solver = SolverFactory("gams:conopt")
result = solver.solve(m, tee=True)

print(result)
for t in pyo.RangeSet(T):
    # utils.report_values(m.period[t])
    print(f'Co/Li recoveries for period {t}')
    print(pyo.value(m.period[t].prec_perc_co))
    print(pyo.value(m.period[t].prec_perc_li))
report_statistics(m)
vals = utils.report_values(m.period[1])
utils.visualize_flows(
    num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals
)

# fix first stage vars
# m.period[1].fs.stage[1].length.fix(m.period[1].fs.stage[1].length.value*2)
# m.period[1].fs.precipitator['permeate'].V.fix(m.period[1].fs.precipitator['permeate'].V.value*2)
# m.period[1].fs.precipitator['retentate'].V.fix(m.period[1].fs.precipitator['retentate'].V.value*2)
# m.period[1].fs.precipitator['retentate'].yields['solvent', 'recycle'].fix()
# m.period[1].fs.precipitator['permeate'].yields['solvent', 'recycle'].fix()
# m.period[1].costing.P_inst.fix(m.period[1].costing.P_inst.value*6)
# m.period[1].fs.stage[1].length.fix(1400)
# m.period[1].fs.precipitator['permeate'].V.fix()
# m.period[1].fs.precipitator['retentate'].V.fix()
# m.period[1].fs.precipitator['retentate'].yields['solvent', 'recycle'].fix()
# m.period[1].fs.precipitator['permeate'].yields['solvent', 'recycle'].fix()
# m.period[1].costing.P_inst.fix(30)
report_statistics(m)

# m.ff = 1
# m.P_co = 0
# m.P_li = 0
# co_rec = []
# li_rec = []
# obj = []

# sweep = np.arange(0, 1.01, .01)
# for i in sweep:
#     m.P_co = i*8000
#     res = solver.solve(m)
#     print(i, ' : ', res.solver.termination_condition)
#     co_rec.append(pyo.value(m.period[1].prec_perc_co)*100)
#     li_rec.append(pyo.value(m.period[1].prec_perc_li)*100)
#     obj.append(pyo.value(m.costing_obj))

# import matplotlib.pyplot as plt
# plt.plot(sweep, co_rec)
# plt.xlabel('Co Price [\$/kg] (-$20 /kg)')
# plt.ylabel('Co Recovery [%]')
# plt.show()

#################################
# uncertainty
#################################

pyros_solver, local_solver, global_solver, first, second, porder \
    = pyros_settings(m, mix)

# deviation from nominal constraint
m.dev = pyo.Param(pyo.RangeSet(T), initialize=1, mutable=True)

for t in pyo.RangeSet(T):
    m.period[t].fs.split_feed.mixed_state[0].flow_vol.unfix()
    m.period[t].fs.split_feed.mixed_state[0].flow_mass_solute['Co'].unfix()
    m.period[t].fs.split_feed.mixed_state[0].flow_mass_solute['Li'].unfix()

@m.Constraint(pyo.RangeSet(T))
def feed_flow_con(b, t):
    """Set uncertainty in feed equality."""
    return m.period[t].fs.split_feed.mixed_state[0].flow_vol == 100*m.dev[t]

@m.Constraint(pyo.RangeSet(T))
def feed_co_con(b, t):
    """Set uncertainty in feed equality."""
    return m.period[t].fs.split_feed.mixed_state[0].flow_mass_solute['Co'] == 1700*m.dev[t]

@m.Constraint(pyo.RangeSet(T))
def feed_li_con(b, t):
    """Set uncertainty in feed equality."""
    return m.period[t].fs.split_feed.mixed_state[0].flow_mass_solute['Li'] == 170*m.dev[t]

# for t in pyo.RangeSet(T):
#     for i in m.period[1].fs.stages:
#         for j in m.period[1].fs.tubes:
#             m.period[t].fs.stage[i].sieving_coefficient['Li', j].unfix()

# @m.Constraint(pyo.RangeSet(T), m.period[1].fs.stages, m.period[1].fs.tubes)
# def li_sieving_con(b, t, i, j):
#     """Set lithium deviations."""
#     return m.period[t].fs.stage[i].sieving_coefficient['Li', j] == 1.3*m.dev[t]

# for t in pyo.RangeSet(T):
#     for i in m.period[1].fs.stages:
#         for j in m.period[1].fs.tubes:
#             m.period[t].fs.stage[i].sieving_coefficient['Co', j].unfix()

# @m.Constraint(pyo.RangeSet(T), m.period[1].fs.stages, m.period[1].fs.tubes)
# def co_sieving_con(b, t, i, j):
#     """Set cobalt deviations."""
#     return m.period[t].fs.stage[i].sieving_coefficient['Co', j] == 0.5*m.dev[t]

solver = SolverFactory("gams:conopt")
result = solver.solve(m, tee=True)

uncparams = [m.dev]
box_set = pyros.BoxSet(bounds=[(0.5, 1.5) for _ in pyo.RangeSet(T)])

#################################
# Increase flow case study
#################################
# for i, j in zip(range(1,T+1), np.concatenate([[100], np.linspace(100,200,T-2), [200]])):
# for i, j in zip(range(1,T+1), np.concatenate([np.flip(np.linspace(50,100,math.ceil(T/2))), np.linspace(50,100,math.floor(T/2))])):
# for i, j in zip(range(1,T+1), np.concatenate([np.linspace(100,150,math.ceil(T/2)), np.flip(np.linspace(100,150,math.floor(T/2)))])):
for i, j in zip(range(1,T+1), [1, 0.7, 0.7, 1, 1, 1.3, 1.3, 1]):
    # print(i,j)
    # m.dev[i] = j/100
    m.dev[i] = j
m.dev.pprint()

solver = pyo.SolverFactory('gams:conopt')
result = solver.solve(m, tee=True)
print(result)
for t in pyo.RangeSet(T):
    # utils.report_values(m.period[t])
    print(f'Co/Li recoveries for period {t}')
    print(pyo.value(m.period[t].prec_perc_co))
    print(pyo.value(m.period[t].prec_perc_li))
report_statistics(m)
NS = num_s
NT = num_t
for t in m.period:
    mod_res = utils.report_values(m.period[t])
    # utils.visualize_flows(NS, NT, conf=mix, model=mod_res, show=False, savefig=True, fname=f'detstudy_period_{t}_{mix}.png')
    utils.visualize_flows(NS, NT, conf=mix, model=mod_res)

# #################################
# # Fouling case study
# #################################
# jinit = 0.1
# m.period[1].fs.stage[2].flux[1].fix(jinit*0.9**1)
# m.period[2].fs.stage[2].flux[1].fix(jinit*0.9**2)
# m.period[2].fs.stage[2].flux[2].fix(jinit*0.9**1)
# m.period[3].fs.stage[2].flux[1].fix(jinit*0.9**3)
# m.period[3].fs.stage[2].flux[2].fix(jinit*0.9**2)
# m.period[3].fs.stage[2].flux[3].fix(jinit*0.9**1)
# m.period[4].fs.stage[2].flux[1].fix(jinit*0.9**4)
# m.period[4].fs.stage[2].flux[2].fix(jinit*0.9**3)
# m.period[4].fs.stage[2].flux[3].fix(jinit*0.9**2)
# m.period[4].fs.stage[2].flux[4].fix(jinit*0.9**1)
# m.period[5].fs.stage[2].flux[1].fix(jinit*0.9**5)
# m.period[5].fs.stage[2].flux[2].fix(jinit*0.9**4)
# m.period[5].fs.stage[2].flux[3].fix(jinit*0.9**3)
# m.period[5].fs.stage[2].flux[4].fix(jinit*0.9**2)
# m.period[5].fs.stage[2].flux[5].fix(jinit*0.9**1)
# m.period[6].fs.stage[2].flux[1].fix(jinit*0.9**6)
# m.period[6].fs.stage[2].flux[2].fix(jinit*0.9**5)
# m.period[6].fs.stage[2].flux[3].fix(jinit*0.9**4)
# m.period[6].fs.stage[2].flux[4].fix(jinit*0.9**3)
# m.period[6].fs.stage[2].flux[5].fix(jinit*0.9**2)
# m.period[6].fs.stage[2].flux[6].fix(jinit*0.9**1)
# m.period[7].fs.stage[2].flux[1].fix(jinit*0.9**7)
# m.period[7].fs.stage[2].flux[2].fix(jinit*0.9**6)
# m.period[7].fs.stage[2].flux[3].fix(jinit*0.9**5)
# m.period[7].fs.stage[2].flux[4].fix(jinit*0.9**4)
# m.period[7].fs.stage[2].flux[5].fix(jinit*0.9**3)
# m.period[7].fs.stage[2].flux[6].fix(jinit*0.9**2)
# m.period[7].fs.stage[2].flux[7].fix(jinit*0.9**1)
# m.period[8].fs.stage[2].flux[1].fix(jinit*0.9**8)
# m.period[8].fs.stage[2].flux[2].fix(jinit*0.9**7)
# m.period[8].fs.stage[2].flux[3].fix(jinit*0.9**6)
# m.period[8].fs.stage[2].flux[4].fix(jinit*0.9**5)
# m.period[8].fs.stage[2].flux[5].fix(jinit*0.9**4)
# m.period[8].fs.stage[2].flux[6].fix(jinit*0.9**3)
# m.period[8].fs.stage[2].flux[7].fix(jinit*0.9**2)
# m.period[8].fs.stage[2].flux[8].fix(jinit*0.9**1)

# solver = pyo.SolverFactory('ipopt')
# result = solver.solve(m, tee=True)
# print(result)
# for t in pyo.RangeSet(T):
#     # utils.report_values(m.period[t])
#     print(f'Co/Li recoveries for period {t}')
#     print(pyo.value(m.period[t].prec_perc_co))
#     print(pyo.value(m.period[t].prec_perc_li))
# report_statistics(m)
# for t in m.period:
#     mod_res = utils.report_values(m.period[t])
#     # utils.visualize_flows(NS, NT, conf=mix, model=mod_res, show=False, savefig=True, fname=f'detstudy_period_{t}_{mix}.png')
#     utils.visualize_flows(NS, NT, conf=mix, model=mod_res, show=True)


#################################
# PyROS
#################################

from pyomo.common.collections import ComponentMap
# Set up stage configuration
var_stages = ComponentMap()
for num, stage in enumerate(second):
    for var in stage:
        var_stages[var] = num + 1

param_stages = ComponentMap((uncparams[0][idx], idx) for idx in uncparams[0])

# add suffixes
m.pyros_separation_priority = pyo.Suffix(direction=pyo.Suffix.LOCAL)
m.pyros_separation_priority[m.li_lb] = 10
m.pyros_separation_priority[m.co_lb] = 10

# solvers

local_solver=pyo.SolverFactory('gams:conopt')
nlp_solvers = [
    pyo.SolverFactory('ipopt'),
    # pyo.SolverFactory('gams:ipopt'),
    # pyo.SolverFactory('gams:minos'),
    # pyo.SolverFactory('gams:snopt'),
]

# instantiate a logger
logger = logging.getLogger("example_pyros_logger")
logger.setLevel(logging.DEBUG)
# add console output handler
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
fh = logging.FileHandler(f"pyros_testrun.log")
fh.setLevel(logging.DEBUG)
logger.addHandler(ch)
logger.addHandler(fh)
# logging.getLogger().setLevel(logging.WARNING)
# solve corresponding RO model with PyROS


# solve
presult = pyros_solver.solve(
    model=m,
    first_stage_variables=first,
    second_stage_variables=second,
    # nested_second_stage_variables=second,
    variable_stages=var_stages,
    uncertain_parameter_stages=param_stages,
    uncertain_params=uncparams,
    multiperiod=True,
    uncertainty_set=box_set,
    local_solver=local_solver,
    global_solver=global_solver,
    backup_local_solvers=nlp_solvers,
    objective_focus=pyros.ObjectiveType.worst_case,
    solve_master_globally=False,
    load_solution=True,
    progress_logger=logger,
    decision_rule_order=1,
    # separation_priority_order=porder,
    bypass_global_separation=True,
    tee=False
)
logger.removeHandler(ch)
logger.removeHandler(fh)

# === Query results ===
time = presult.time
iterations = presult.iterations
termination_condition = presult.pyros_termination_condition
objective = presult.final_objective_value
# === Print some results ===
single_stage_final_objective = objective
print(f"Final objective value: {single_stage_final_objective}")
print(f"PyROS termination condition: {termination_condition}")
print(f"Time: {time}")
worstcase = max(presult.model_data.master_results.master_model.scenarios.keys(),key=lambda idx: pyo.value(presult.model_data.master_results.master_model.scenarios[idx].second_stage_objective), )
alldrvars = []
alldrvars_names = []
for i in presult.model_data.master_results.master_model.scenarios[worstcase].second_stage.decision_rule_eqns:
    drvars = []
    drvars_names = []
    for term in presult.model_data.master_results.master_model.scenarios[worstcase].second_stage.decision_rule_eqns[i].body.args:
        try:
            # print(term.args[1])
            drvars.append(pyo.value(term.args[1]))
            drvars_names.append(term.args[1].name)
        except:
            # print(term)
            drvars.append(pyo.value(term))
            drvars_names.append(term.name)
    alldrvars.append(drvars)
    alldrvars_names.append(drvars_names)

alldrvars = np.array(alldrvars)
print(np.round(alldrvars[:, :-1]))

# for t in pyo.RangeSet(T): utils.report_values(m.period[t])




# rdevs = [0.2]
# # rdevs = [0.03, 0.06, 0.09, 0.12, 0.15, 0.18, 0.2]
# # rdevs = [0.12, 0.13, 0.14, 0.15]
# # relative_deviation = 0.3

# ells = []
# elvl = [0.99]
# # elvl = np.arange(0.1, 1, 0.3)
# # uncset, uncertain_parameters = unc_setup(m, relative_deviation, 1)
# for relative_deviation in rdevs:
#     ellipsoids, uncparams = construct_confidence_ellipsoids(
#         m,
#         relative_deviation,
#         0.75,
#         conf_lvls=elvl,
#         loc={
#             'stage': [1, 2, 3],
#             'tube': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
#         },
#     )
#     ells.append(ellipsoids)

# ellipsoids = {}
# for i in range(len(ells)):
#     ellipsoids[rdevs[i]] = ells[i][0.99]

# # instantiate a logger
# logger = logging.getLogger("example_pyros_logger")
# logger.setLevel(logging.INFO)

# # add console output handler
# ch = logging.StreamHandler()
# logger.addHandler(ch)
# # logging.getLogger().setLevel(logging.WARNING)


# nlp_solvers = [
#     pyo.SolverFactory('gams:conopt'),
#     # pyo.SolverFactory('gams:ipopt'),
#     # pyo.SolverFactory('gams:minos'),
#     # pyo.SolverFactory('gams:snopt'),
#         ]

# # expected confidence levels:
# count = 0

# # track design choices
# designs = {
#     'deviation': [],
#     'area': [],
#     'pump': [],
#     'ret_precip': [],
#     'perm_precip': [],
#     'cost': []
# }
# designs['deviation'].append(0)
# designs['area'].append(pyo.value(m.fs.stage[1].length))
# designs['pump'].append(pyo.value(m.costing.P_inst))
# designs['ret_precip'].append(pyo.value(m.fs.precipitator['retentate'].V))
# designs['perm_precip'].append(pyo.value(m.fs.precipitator['permeate'].V))
# designs['cost'].append(pyo.value(m.costing.obj))

# # solve RO model subject to each confidence ellipoid
# for lvl, ellipsoid in ellipsoids.items():
#     print(f'***{lvl}***')
#     print(ellipsoid.center)
#     print(ellipsoid.shape_matrix)
#     print(ellipsoid.scale)
#     # solve corresponding RO model with PyROS
#     presult = pyros_solver.solve(
#         model=m,
#         first_stage_variables=first,
#         second_stage_variables=second,
#         uncertain_params=uncparams,
#         uncertainty_set=ellipsoid,
#         local_solver=local_solver,
#         global_solver=global_solver,
#         backup_local_solvers=nlp_solvers,
#         objective_focus=pyros.ObjectiveType.worst_case,
#         solve_master_globally=False,
#         load_solution=True,
#         progress_logger=logger,
#         decision_rule_order=0,
#         separation_priority_order=porder,
#         bypass_global_separation=True,
#         tee=False
#     )

#     # === Query results ===
#     time = presult.time
#     iterations = presult.iterations
#     termination_condition = presult.pyros_termination_condition
#     objective = presult.final_objective_value
#     # === Print some results ===
#     single_stage_final_objective = objective
#     print(f"Final objective value: {single_stage_final_objective}")
#     print(f"PyROS termination condition: {termination_condition}")
#     print(f"Time: {time}")
#     # utils.report_values(m)
#     utils.report_cost(m)
#     mod_res = utils.report_values(m)
#     # utils.visualize_flows(NS, NT, conf=mix, model=mod_res)

#     worst_obj_idx= max(presult.pyros_soln.master_results.master_model.scenarios.keys(), key=lambda idx: pyo.value(presult.pyros_soln.master_results.master_model.scenarios[idx].second_stage_objective),)
#     obj_res = list(pyo.value(presult.pyros_soln.master_results.master_model.scenarios[idx].second_stage_objective) for idx in presult.pyros_soln.master_results.master_model.scenarios.keys())
#     print(worst_obj_idx)
#     print(obj_res)
#     count += 1
#     print(count)
#     print(pyo.value(m.fs.stage[1].length))
#     print(pyo.value(m.costing.P_inst))
#     print(pyo.value(m.fs.precipitator['retentate'].V))
#     print(pyo.value(m.fs.precipitator['permeate'].V))
#     print(pyo.value(m.costing.obj))

#     if presult.pyros_termination_condition.name == 'robust_feasible':
#         designs['deviation'].append(lvl)
#         designs['area'].append(pyo.value(m.fs.stage[1].length))
#         designs['pump'].append(pyo.value(m.costing.P_inst))
#         designs['ret_precip'].append(pyo.value(m.fs.precipitator['retentate'].V))
#         designs['perm_precip'].append(pyo.value(m.fs.precipitator['permeate'].V))
#         designs['cost'].append(pyo.value(m.costing.obj))
#         model_solutions[f'robust_{lvl}'] = {
#             var.name: pyo.value(var)
#             for var in m.component_data_objects(pyo.Var)
#         }
#         # utils.visualize_flows(
#         #     NS,
#         #     NT,
#         #     conf=mix,
#         #     model=mod_res,
#         #     show=False,
#         #     savefig=True,
#         #     fname=f'high-area_sieving_robust_{lvl}_SRO.png'
#         # )
#         # print('*'*8)
#         # print('Area')
#         # curr_rec_val = pyo.value(m.fs.stage[1].length)
#         # print('Current Deterministic Value: ', curr_rec_val)
#         # for s in m.fs.stages:
#         #     #     print(f'Current UB Stage {s}: ', pyo.value(m.fs.stage[s].length.upper))
#         #     print(f'Current LB Stage {s}: ', pyo.value(m.fs.stage[s].length.lower))
#         # # print('Set RO UB: ', curr_rec_val*(1 + 1.4))
#         # print('Set RO LB: ', curr_rec_val+30)
#         # for s in m.fs.stages:
#         #     #     m.fs.stage[s].length.setub(curr_rec_val*(1 + 1.4))
#         #     m.fs.stage[s].length.setlb(curr_rec_val+30)
#     else:
#         designs['deviation'].append(lvl)
#         designs['area'].append(0)
#         designs['pump'].append(0)
#         designs['ret_precip'].append(0)
#         designs['perm_precip'].append(0)
#         designs['cost'].append(0)



#     # # reset recycle UB
#     # m.fs.split_diafiltrate.inlet.flow_vol.setub(1000)
#     # for s in m.fs.stages:
#     #     m.fs.stage[s].length.setub(2500)


# logger.removeHandler(ch)

# # import pandas as pd
# # # save results
# # data = pd.DataFrame(
# #     designs
# # )
# # data.to_csv(f'new_CoR_sieving_designs_ARO_{mix}_{NS,NT}.csv')

# # import json
# # with open(f'new_CoR_sieving_models_ARO_{mix}_{NS, NT}.json', 'w') as f:
# #     json.dump(model_solutions, f)

