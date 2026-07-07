from pyros_heatmap.heatmap import evaluate_heatmap
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

    return model

def ellipsoid_probability(x, mean, cov_mat):
    """
    Evaluate Gaussian probability density function
    parameterized by a given mean and covariance matrix.
    """
    n = len(mean)
    assert cov_mat.shape == (n, n)

    # convert to arrays
    x = np.array(x)
    mean = np.array(mean)
    cov_mat = np.array(cov_mat)

    # now evaluate the probability
    denom = (2 * np.pi) ** (n / 2) * np.linalg.det(cov_mat) ** (1 / 2)
    return (1 / denom) * np.exp(
        -(1 / 2)
        ,* (x - mean).reshape(1, n)
        @ np.linalg.inv(cov_mat)
        @ (x - mean)
    )

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
m.fs.lithium_carbonate_price = 0
m.fs.cobalt_oxalate_price = 0
m.recovery_li = 0.7
m.recovery_co = 0.7

solver = SolverFactory("ipopt")
# solver = SolverFactory('multistart_solve')
# solver.mix = mix_style
# solver.dsolver = 'gams:conopt'

result = solver.solve(m, tee=False)
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

deterministic = {}
deterministic['deterministic'] = {
    var.name: pyo.value(var)
    for var in m.component_data_objects(pyo.Var)
}

# # update recycle UB for PyROS solves
# print('*'*8)
# print('Recycles')
# curr_rec_val = pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0])
# print('Current Deterministic Value: ', curr_rec_val)
# print('Current UB: ', pyo.value(m.fs.split_diafiltrate.inlet.flow_vol[0].upper))
# print('Set RO UB: ', curr_rec_val*3.5)
# m.fs.split_diafiltrate.inlet.flow_vol.setub(curr_rec_val*3.5)
# print('*'*8)
# print('Area')
# curr_rec_val = pyo.value(m.fs.stage[1].length)
# print('Current Deterministic Value: ', curr_rec_val)
# for s in m.fs.stages:
#     print(f'Current UB Stage {s}: ', pyo.value(m.fs.stage[s].length.upper))
# print('Set RO UB: ', curr_rec_val*3.5)
# for s in m.fs.stages:
#     m.fs.stage[s].length.setub(curr_rec_val*3.5)




pyros_solver, local_solver, global_solver, first, second, porder \
    = pyros_settings(m, mix)

rdevs = [0.1, 0.2, 0.3, 0.4, 0.5]
rdevs = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
rdevs = [0.3]
# relative_deviation = 0.3

ells = []
elvl = [0.99]
# elvl = np.arange(0.1, 1, 0.3)
# uncset, uncertain_parameters = unc_setup(m, relative_deviation, 1)
for relative_deviation in rdevs:
    ellipsoids, uncparams = construct_confidence_ellipsoids(
        m,
        relative_deviation,
        0.75,
        conf_lvls=elvl,
        loc={
            'stage': [1, 2, 3],
            'tube': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        },
    )
    ells.append(ellipsoids)

ellipsoids = {}
for i in range(len(ells)):
    ellipsoids[rdevs[i]] = ells[i][0.99]


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
#         decision_rule_order=1,
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

#     # # reset recycle UB
#     # m.fs.split_diafiltrate.inlet.flow_vol.setub(1000)
#     # for s in m.fs.stages:
#     #     m.fs.stage[s].length.setub(2500)


# logger.removeHandler(ch)
# logging.basicConfig(level=logging.INFO)
# model_solutions['robust'] = {
#     var.name: pyo.value(var)
#     for var in m.component_data_objects(pyo.Var)
# }

logging.basicConfig(level=logging.INFO)
import json
with open('test_sieve_newmodel_models_ARO_stage.json', 'r') as f:
    model_solutions = json.load(f)
temp = {
    'Deterministic': model_solutions['deterministic'],
    'PyROS Robust Design': model_solutions['robust_0.3'],
}
model_solutions = temp

# project into 2D space for each ellipsoid
out_loc = f"./test_sieve_newmodel_conopt"
for i in range(1):
    axis = [2*i, 2*i+1]
    # proj_ellipsoids = construct_projected_confidence_ellipsoids(ellipsoids, axis)
    proj_ellipsoids = ellipsoids
    print(proj_ellipsoids)
    proj_uncparams = [uncparams[i] for i in axis]
    for i in proj_uncparams: print(i.name)

    # evaluate the heatmap for all solutions
    # highlight_set_idxs = {"deterministic": []}
    # highlight_set_idxs.update(
    #     {
    #         name: [idx]
    #         for idx, name in enumerate([f"{100 * lvl:.2f}% confidence RO" for lvl in proj_ellipsoids.keys()])
    #     }
    # )
    highlight_set_idxs = {"Deterministic": []}
    highlight_set_idxs.update(
        {
            name: [idx - 1]
            for idx, name in enumerate(model_solutions.keys())
            # for idx, name in enumerate(['deterministic',
            #                             'Robust to $\\pm$10%\nDeviation',
            #                             'Robust to $\\pm$20%\nDeviation',
            #                             'Robust to $\\pm$30%\nDeviation',
            #                             'Robust to $\\pm$40%\nDeviation',
            #                             'Robust to $\\pm$50%\nDeviation'])
            if idx >= 1
        }
    )
    hm_res = evaluate_heatmap(
        model=m,
        first_stage_vars=first,
        uncertain_params=proj_uncparams,
        uncertainty_sets=list(proj_ellipsoids.values()),
        solver=pyo.SolverFactory('gams:conopt'),
        model_solutions=list(model_solutions.values()),
        solution_loader=load_sol,
        solution_names=[key for key in model_solutions.keys()],
        uncertainty_set_names=[
            f"{100 * lvl}% Deviation" for lvl in proj_ellipsoids.keys()
        ],
        # highlight_set_idxs=highlight_set_idxs,
        discretize_set_idx=-1,
        output_dir=out_loc,
        output_plots=True,
        output_results=True,
        output_problems=False,
        output_logs=False,
        output_problem_types=[".gms", ".bar", ".nl"],
        grid_resolution=13,
        param_plot_axis_names={
            proj_uncparams[0].name: f"$S_{{Li}}$",
            proj_uncparams[1].name: f"$S_{{Co}}$",
        },
        sampling_mode="uniform_random",
        num_sampling_points=250,
        rng=np.random.default_rng(seed=7),
        # weight_func=ellipsoid_probability,
        # weight_func_kwargs=dict(
        #     mean=list(proj_ellipsoids.values())[-1].center,
        #     cov_mat=list(proj_ellipsoids.values())[-1].shape_matrix,
        # ),
        cust_nom_val=1e6,
        fixed_con_feas_tol=1e-6
    )
    # cb = hm_res[2].axes[-1]
    # cb.set_ylabel('TAC [$\\frac{k\\$}{yr}$]', fontsize=14)
    # hm_res[2].savefig('rtr_CoR_sieving_heatmap_ARO_simple_(3, 10).png')
