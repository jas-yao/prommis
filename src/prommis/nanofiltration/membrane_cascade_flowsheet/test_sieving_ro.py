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
from multistart_solve import MultistartSolve
from matplotlib import pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition, SolverResults
from idaes.core.util.model_statistics import report_statistics
import utils
import pyomo.contrib.pyros as pyros
from pyros_cost_settings import pyros_settings
import logging
import numpy as np
from small_tilt_ell_unc_setup import unc_setup, construct_cov_mat
import confidence_ellipsoid.confidence_ellipsoid as ce
from multistart_solve import MultistartSolve

import logging

import pyomo.environ as pyo
import pyomo.contrib.pyros as pyros
import numpy as np

from pyomo.contrib.fbbt.fbbt import fbbt

def load_sol(model, sol):
    """
    Load solution (dict if variable names and values)
    to model.
    """
    for key, val in sol.items():
        model.find_component(key).set_value(val)

    model.R.pprint()
    model.Rco.pprint()
    return model

def construct_confidence_ellipsoids(m, p, k, conf_lvls, loc):
    """
    Construct confidence ellipsoids for the reactor-separator
    model.
    """
    n = len(m.fs.stages)*len(m.fs.tubes)*2
    confidence_level_scales = dict()
    for lvl in conf_lvls:
        confidence_level_scales[lvl] = ce.mag_factor(lvl, 2)
    print(confidence_level_scales)


    confidence_ellipsoids = dict()
    # solve RO model subject to each confidence ellipoid
    for lvl, scale in confidence_level_scales.items():
        uncset, uncparams = unc_setup(m, p, k, scale, loc, indiv=False)
        confidence_ellipsoids[lvl] = uncset

    return confidence_ellipsoids, uncparams

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
m.fs.split_diafiltrate.inlet.flow_vol.setlb(1e-11)
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

# set recovery lower bounds
m.fs.soda_ash_price = 0
m.fs.ammonium_oxalate_price = 0
m.fs.lithium_carbonate_price = 0
m.fs.cobalt_oxalate_price = 0
# m.recovery_li = 0
# m.recovery_co = 0

# # solve for initial feasible solution
# solver = SolverFactory("ipopt")
# result = solver.solve(m, tee=False)

# # tighten bounds
# fbbt(m)

m.recovery_li = 0.2
m.recovery_co = 0.7

# solver = SolverFactory("ipopt")
# solver = SolverFactory("gams:conopt")
# result = solver.solve(m, tee=False)
# m.recovery_li = 0.5
# m.recovery_co = 0.5
solver = SolverFactory('multistart_solve')
solver.mix = mix_style
solver.dsolver = 'gams:conopt'

result = solver.solve(m, tee=False)
# result = solver.solve(m, tee=True)
assert_optimal_termination(result)

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


model_solutions = {}
model_solutions['deterministic'] = {
    var.name: pyo.value(var)
    for var in m.component_data_objects(pyo.Var)
}


# # update recycle UB for PyROS solves
# print('*'*8)
# print('Recycles')
# # curr_rec_val = pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0])
# curr_rec_val = pyo.value(m.costing.P_inst)
# print('Current Deterministic Value: ', curr_rec_val)
# print('Current UB: ', pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0].upper))
# # print('Current LB: ', pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0].lower))
# print('Set RO UB: ', 100)
# # # print('Set RO LB: ', curr_rec_val*(1 - 1.4))
# m.costing.P_inst.setub(100)
# # # m.fs.split_diafiltrate.inlet.flow_vol.setlb(curr_rec_val*(1 - 1.4))
# print('*'*8)
# print('Area')
# curr_rec_val = pyo.value(m.fs.stage[1].length)
# print('Current Deterministic Value: ', curr_rec_val)
# for s in m.fs.stages:
# #     print(f'Current UB Stage {s}: ', pyo.value(m.fs.stage[s].length.upper))
#     # print(f'Current LB Stage {s}: ', pyo.value(m.fs.stage[s].length.lower))
#     print(f'Current LB Stage {s}: ', pyo.value(m.fs.stage[s].length.lower))
# # print('Set RO UB: ', curr_rec_val*(1 + 1.4))
# # print('Set RO LB: ', 810)
# print('Set RO UB: ', 470)
# for s in m.fs.stages:
# #     m.fs.stage[s].length.setub(curr_rec_val*(1 + 1.4))
#       # m.fs.stage[s].length.setlb(810)
#       m.fs.stage[s].length.setub(470)

# # bound tightening with fbbt
# print('*'*8)
# print('*Recycles*')
# print('Current UB: ', pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0].upper))
# print('Current LB: ', pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0].lower))
# print('*Area*')
# for s in m.fs.stages:
#     print(f'Current UB Stage {s}: ', pyo.value(m.fs.stage[s].length.upper))
#     print(f'Current LB Stage {s}: ', pyo.value(m.fs.stage[s].length.lower))
# print('Tightening bounds with FBBT')
# fbbt(m)
# print('*Recycles*')
# print('Current UB: ', pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0].upper))
# print('Current LB: ', pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0].lower))
# print('*Area*')
# for s in m.fs.stages:
#     print(f'Current UB Stage {s}: ', pyo.value(m.fs.stage[s].length.upper))
#     print(f'Current LB Stage {s}: ', pyo.value(m.fs.stage[s].length.lower))
# print('*'*8)


# pyros_solver, local_solver, global_solver, first, second, porder \
#     = pyros_settings(m, mix)

# rdevs = [0.1]
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


# nlp_solvers = [
#     # pyo.SolverFactory('gams:conopt'),
#     pyo.SolverFactory('ipopt')
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
# designs['pump'].append(pyo.value(m.fs.diafiltrate_pump.costing.pump_power)/8766)
# designs['ret_precip'].append(pyo.value(m.fs.precipitator['retentate'].volume))
# designs['perm_precip'].append(pyo.value(m.fs.precipitator['permeate'].volume))
# designs['cost'].append(pyo.value(m.cost_objective))

# # solve RO model subject to each confidence ellipoid
# for lvl, ellipsoid in ellipsoids.items():
#     print(f'***{lvl}***')
#     print(ellipsoid.center)
#     print(ellipsoid.shape_matrix)
#     print(ellipsoid.scale)
#     # instantiate a logger
#     logger = logging.getLogger("example_pyros_logger")
#     logger.setLevel(logging.DEBUG)
#     # add console output handler
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.INFO)
#     fh = logging.FileHandler(f"debug_sieving_unc_newmodel.log")
#     fh.setLevel(logging.DEBUG)
#     logger.addHandler(ch)
#     logger.addHandler(fh)
#     # logging.getLogger().setLevel(logging.WARNING)
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
#         decision_rule_order=1,
#         separation_priority_order=porder,
#         bypass_global_separation=True,
#         tee=False
#     )
#     logger.removeHandler(ch)
#     logger.removeHandler(fh)

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
#     # utils.report_cost(m)
#     mod_res = utils.report_values(m)
#     # utils.visualize_flows(NS, NT, conf=mix, model=mod_res)

#     # worst_obj_idx= max(presult.pyros_soln.master_results.master_model.scenarios.keys(), key=lambda idx: pyo.value(presult.pyros_soln.master_results.master_model.scenarios[idx].second_stage_objective),)
#     # obj_res = list(pyo.value(presult.pyros_soln.master_results.master_model.scenarios[idx].second_stage_objective) for idx in presult.pyros_soln.master_results.master_model.scenarios.keys())
#     # print(worst_obj_idx)
#     # print(obj_res)
#     count += 1
#     print(count)
#     print(pyo.value(m.fs.stage[1].length))
#     print(pyo.value(m.fs.diafiltrate_pump.costing.pump_power)/8766)
#     print(pyo.value(m.fs.precipitator['retentate'].volume))
#     print(pyo.value(m.fs.precipitator['permeate'].volume))
#     print(pyo.value(m.cost_objective))

#     if presult.pyros_termination_condition.name == 'robust_feasible':
#         designs['deviation'].append(lvl)
#         designs['area'].append(pyo.value(m.fs.stage[1].length))
#         designs['pump'].append(pyo.value(m.fs.diafiltrate_pump.costing.pump_power)/8766)
#         designs['ret_precip'].append(pyo.value(m.fs.precipitator['retentate'].volume))
#         designs['perm_precip'].append(pyo.value(m.fs.precipitator['permeate'].volume))
#         designs['cost'].append(pyo.value(m.cost_objective))
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
#         vals = utils.report_values(m)
#         utils.visualize_flows(
#             num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals
#         )
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



# import pandas as pd
# # save results
# data = pd.DataFrame(
#     designs
# )
# # data.to_csv(f'test_sieve_newmodel_designs_ARO_{mix}.csv')

# # import json
# # with open(f'test_sieve_newmodel_models_ARO_{mix}.json', 'w') as f:
# #     json.dump(model_solutions, f)
