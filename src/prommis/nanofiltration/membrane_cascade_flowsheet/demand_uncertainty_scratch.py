#####################################################################################################
# “PrOMMiS” was produced under the DOE Process Optimization and Modeling for Minerals Sustainability
# (“PrOMMiS”) initiative, and is copyright (c) 2023-2025 by the software owners: The Regents of the
# University of California, through Lawrence Berkeley National Laboratory, et al. All rights reserved.
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license information.
#####################################################################################################
"""Scratch file for testing demand uncertainty."""

from pyomo.environ import (
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    value,
)
import pyomo.environ as pyo

from idaes.core.util import to_json, from_json
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import report_statistics

from prommis.nanofiltration.membrane_cascade_flowsheet import utils
from prommis.nanofiltration.membrane_cascade_flowsheet.diafiltration_flowsheet_model import (
    DiafiltrationModel,
)

# from multistart_solve import MultistartSolve
import logging
import numpy as np

######################################################################
# Deterministic model setup
######################################################################

mix_style = "stage"


def solve_scaled_model(m, L, C):
    m.recovery_li = L
    m.recovery_co = C

    scaling = TransformationFactory("core.scale_model")
    solver = SolverFactory("gams:conopt")
    # solver = SolverFactory('multistart_solve')
    # solver.mix = mix_style
    # solver.dsolver = 'gams:conopt'
    # solver._subsolver.options["add_options"] = [
    #     "option reslim=20;",
    # ]

    scaled_model = scaling.create_using(m, rename=False)
    try:
        result = solver.solve(scaled_model, tee=True)
    except:
        print("ValueError: Cannot load a SolverResults object with bad status: error")
        result = None
    # assert_optimal_termination(result)
    # try:
    #     assert_optimal_termination(result)
    # except:
    #     result = backup_solver.solve(scaled_model, tee=False)
    #     assert_optimal_termination(result)
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
lithium_recovery = 0.5
cobalt_recovery = 0.75
m.fs.lithium_carbonate_price = 0
m.fs.cobalt_oxalate_price = 0

result = solve_scaled_model(
    m,
    L=lithium_recovery,
    C=cobalt_recovery,
)
print(result.solver.termination_condition)

dt = DiagnosticsToolbox(m)
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
utils.visualize_flows(num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals)

######################################################################
# Discrete scenario sampling for uncertain demand
######################################################################

# latin hypercube sampling
from scipy.stats import qmc

n = 5
sample_scaled = []
temp = n

while len(sample_scaled) < n:
    sampler = qmc.LatinHypercube(d=2, seed=7)
    sample = sampler.random(n=temp)
    l_bounds = [0.1, 0.1]
    u_bounds = [0.81, 0.89]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    # sample_scaled = np.append([[m.recovery_li.value, m.recovery_co.value]],sample_scaled, axis=0)
    # sample_scaled = np.append([[0.8, 0.45]],sample_scaled, axis=0)
    # sample_scaled = np.append([[0.4, 0.85]],sample_scaled, axis=0)
    sample_scaled = np.append([[0.5, 0.75]], sample_scaled, axis=0)
    sample_scaled = np.array([i for i in sample_scaled[:, :] if i[0] + i[1] <= 1.25])
    # sample_scaled = np.array([[0.5, 0.6], [0.6, 0.5], [0.5,0.5]])
    print(len(sample_scaled))
    if len(sample_scaled) >= n:
        sample_scaled = sample_scaled[:n]
    temp += 1

# fix first stage:
# m.fs.stage[1].length.fix(813.6209)
# m.fs.diafiltrate_pump.costing.install_inlet_vol_flow.fix(69.1917)
# m.fs.feed_pump.costing.install_inlet_vol_flow.fix(100)
# m.fs.precipitator['retentate'].volume.fix(95.6941)
# m.fs.precipitator['permeate'].volume.fix(99.751)
# m.fs.precipitator['retentate'].yields['solvent', 'recycle'].fix(0)
# m.fs.precipitator['permeate'].yields['solvent', 'recycle'].fix(0)
m.fs.stage[1].length.fix()
m.fs.diafiltrate_pump.costing.install_inlet_vol_flow.fix()
m.fs.feed_pump.costing.install_inlet_vol_flow.fix()
m.fs.precipitator['retentate'].volume.fix()
m.fs.precipitator['permeate'].volume.fix()
m.fs.precipitator['retentate'].yields['solvent', 'recycle'].fix()
m.fs.precipitator['permeate'].yields['solvent', 'recycle'].fix()
# l1 = m.fs.stage[1].length.value
# l2 = m.fs.diafiltrate_pump.costing.install_inlet_vol_flow.value
# l3 = m.fs.feed_pump.costing.install_inlet_vol_flow.value
# l4 = m.fs.precipitator['retentate'].volume.value
# l5 = m.fs.precipitator['permeate'].volume.value
# l6 = m.fs.precipitator['retentate'].yields['solvent', 'recycle'].value
# l7 = m.fs.precipitator['permeate'].yields['solvent', 'recycle'].value
# m.fs.stage[1].length.fix(l1*1.1)
# m.fs.diafiltrate_pump.costing.install_inlet_vol_flow.fix(l2 * 1.1)
# m.fs.feed_pump.costing.install_inlet_vol_flow.fix(l3 * 1.1)
# m.fs.precipitator["retentate"].volume.fix(l4 * 1.1)
# m.fs.precipitator["permeate"].volume.fix(l5 * 1.1)
# m.fs.precipitator["retentate"].yields["solvent", "recycle"].fix(l6 * 1.1)
# m.fs.precipitator["permeate"].yields["solvent", "recycle"].fix(l7 * 1.1)
# test_solver = SolverFactory('gams:conopt')
# res = test_solver.solve(m)
# print(res)

costs = []
sample_scaled = sample_scaled[::-1]
for i in sample_scaled:
    print(i)
    result = solve_scaled_model(
        m,
        L=i[0],
        C=i[1],
    )
    # m.recovery_li = i[0]
    # m.recovery_co = i[1]
    # solver = SolverFactory('gams:conopt')
    # solver.options["add_options"] = [
    #     "option reslim=30;",
    # ]
    # # result = solver.solve(m)
    # try:
    #     result = solver.solve(m, tee=False)
    #     print(result.solver.termination_condition)
    # except:
    #     print('ValueError: Cannot load a SolverResults object with bad status: error')
    #     result= None
    try:
        assert_optimal_termination(result)
        print(result.solver.termination_condition)
        costs.append(value(m.cost_objective))
    except:
        costs.append(np.nan)


# plot samples in 2D space
import matplotlib.pyplot as plt

pyros_points = np.array(
    [
        [0.5, 0.75],
        [0.79716581, 0.26461255],
        [0.62807737, 0.24361373],
        [0.75282269, 0.4924542],
        [0.53015778, 0.5723761],
        [0.48236574, 0.66438673],
        [0.4292196, 0.19180233],
        [0.29487116, 0.12052478],
        [0.72116759, 0.44863844],
        [0.74583765, 0.46127366],
        [0.4231216, 0.62129423],
        [0.11629223, 0.2219565],
        [0.38805974, 0.7006465],
        [0.38066511, 0.78539242],
        [0.26219454, 0.86796327],
        [0.10795618, 0.36890236],
        [0.14742721, 0.64152552],
        [0.20677129, 0.8411622],
    ]
)

fig, ax = plt.subplots()
cmap = plt.cm.viridis
norm = plt.Normalize(np.nanmin(costs) / 1e6, np.nanmax(costs) / 1e6)

# Plot each point with its own color
for i, (li, co, cost) in enumerate(
    zip(sample_scaled[:, 0] * 100, sample_scaled[:, 1] * 100, np.array(costs) / 1e6)
):
    if not np.isnan(cost):
        plt.plot(
            li,
            co,
            "x",
            color=cmap(norm(cost)),
            marker=".",
            markersize=12,
            markeredgewidth=5,
            linestyle="None",
        )
    else:
        plt.plot(
            li,
            co,
            "x",
            color="red",
            marker="x",
            markersize=12,
            markeredgewidth=2,
            linestyle="None",
            fillstyle="none",
            label="Infeasible" if i == 0 else "",
        )

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label="Annualized Cost $MM")
cbar.set_label("Annualized Cost [$MM]", fontsize=12)

# plt.plot(sample_scaled[:,0]*100, sample_scaled[:,1]*100, marker='x', markersize=12, markeredgewidth=5, linestyle='None')
# plt.plot(pyros_points[:,0]*100, pyros_points[:,1]*100, marker='*', markersize=6, markeredgewidth=3, linestyle='None', color='red', label='PyROS Sampled Point')
# plt.plot(pyros_points[14,0]*100, pyros_points[14,1]*100, marker='o', markersize=16, markeredgewidth=2, linestyle='None', fillstyle='none', markeredgecolor='red', label='Worst-Case')
plt.xlim([0, 81])
plt.ylim([0, 89])
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.tick_params(direction="in", top=True, right=True)
plt.xlabel("Lithium Recovery [%]", fontsize=16, fontweight="bold")
plt.ylabel("Cobalt Recovery [%]", fontsize=16, fontweight="bold")
plt.grid()

# ax.legend(loc='upper center',
#           bbox_to_anchor=(0.5, -0.15),
#           ncol=1,
#           frameon=False)
ax.legend(loc="lower left")
plt.savefig("milestone_robust_heatmap.png", dpi=600, bbox_inches="tight")
plt.show()

# discrete uncertainty set
import pyomo.contrib.pyros as pyros

discrete_set = pyros.DiscreteScenarioSet(scenarios=sample_scaled)


######################################################################
# PyROS
######################################################################

# partition variables
first = [
    m.fs.stage[1].length,
    m.fs.diafiltrate_pump.costing.install_inlet_vol_flow,
    m.fs.feed_pump.costing.install_inlet_vol_flow,
    m.fs.precipitator["retentate"].volume,
    m.fs.precipitator["permeate"].volume,
    m.fs.precipitator["retentate"].yields["solvent", "recycle"],
    m.fs.precipitator["permeate"].yields["solvent", "recycle"],
]
second = [
    m.fs.split_diafiltrate.mixed_state[0].flow_vol,
    m.fs.precipitator["retentate"].split_inlet["bypass"],
    m.fs.precipitator["permeate"].split_inlet["bypass"],
]
first, second = utils.sep_dof(m, mix_style, first, second)
print("Number of First Stage Vars:", len(first))
print("Number of Second Stage Vars:", len(second))

uncparams = [m.recovery_li, m.recovery_co]

# add suffixes
m.pyros_separation_priority = pyo.Suffix(direction=pyo.Suffix.LOCAL)
m.pyros_separation_priority[m.prec_li_lb] = 10
m.pyros_separation_priority[m.prec_co_lb] = 10
porder = {}
porder["prec_li_lb"] = 1
porder["prec_co_lb"] = 1

# solvers
local_solver = pyo.SolverFactory("ipopt")
global_solver = pyo.SolverFactory("ipopt")
pyros_solver = pyo.SolverFactory("pyros")

nlp_solvers = [
    # pyo.SolverFactory('ipopt'),
    # pyo.SolverFactory('gams:conopt'),
    # pyo.SolverFactory('gams:ipopt'),
    # pyo.SolverFactory('gams:minos'),
    # pyo.SolverFactory('gams:snopt'),
]

# instantiate a logger
logger = logging.getLogger("example_pyros_logger")
logger.setLevel(logging.DEBUG)

# add console output handler
ch = logging.StreamHandler()
logger.addHandler(ch)

# solve
presult = pyros_solver.solve(
    model=m,
    first_stage_variables=first,
    second_stage_variables=second,
    # nested_second_stage_variables=second,
    uncertain_params=uncparams,
    # multiperiod=True,
    uncertainty_set=discrete_set,
    local_solver=local_solver,
    global_solver=global_solver,
    backup_local_solvers=nlp_solvers,
    objective_focus=pyros.ObjectiveType.worst_case,
    solve_master_globally=False,
    load_solution=True,
    progress_logger=logger,
    decision_rule_order=1,
    separation_priority_order=porder,
    bypass_global_separation=True,
    tee=False,
)
logger.removeHandler(ch)

# === Query results ===
time = presult.time
iterations = presult.iterations
termination_condition = presult.pyros_termination_condition
objective = presult.final_objective_value
# === Print some results ===
# single_stage_final_objective = objective
# print(f"Final objective value: {single_stage_final_objective}")
# print(f"PyROS termination condition: {termination_condition}")
# print(f"Time: {time}")
# worstcase = max(presult.model_data.master_results.master_model.scenarios.keys(),key=lambda idx: pyo.value(presult.model_data.master_results.master_model.scenarios[idx].second_stage_objective), )
# alldrvars = []
# for i in presult.model_data.master_results.master_model.scenarios[worstcase].second_stage.decision_rule_eqns:
#     drvars = []
#     for term in presult.model_data.master_results.master_model.scenarios[worstcase].second_stage.decision_rule_eqns[i].body.args:
#         try:
#             drvars.append(pyo.value(term.args[1]))
#         except:
#             pass
#     alldrvars.append(drvars)

# alldrvars = np.array(alldrvars)
# print(alldrvars[:, :-1])
