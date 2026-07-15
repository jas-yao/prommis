"""Disturbed Flow Case Study."""
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
from prommis.nanofiltration.membrane_cascade_flowsheet.pyros_setting import pyros_settings
import logging
import numpy as np
# from small_tilt_ell_unc_setup import unc_setup, construct_cov_mat
import confidence_ellipsoid.confidence_ellipsoid as ce

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
# utils.visualize_flows(
#     num_boxes=num_s, num_sub_boxes=num_t, conf=mix_style, model=vals
# )

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
# report_statistics(m)

#################################
# uncertainty
#################################

# pyros setup
pyros_solver, local_solver, global_solver, first, second, porder \
    = pyros_settings(m, mix)

def add_multiperiod_unc_con(m, choice='flow'):
    # deviations from nominal constraint
    m.dev = pyo.Param(pyo.RangeSet(T), initialize=1, mutable=True)

    if choice == 'flow':
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

        for t in pyo.RangeSet(T):
            m.period[t].fs.split_feed.mixed_state[0].flow_vol.unfix()
            m.period[t].fs.split_feed.mixed_state[0].flow_mass_solute['Co'].unfix()
            m.period[t].fs.split_feed.mixed_state[0].flow_mass_solute['Li'].unfix()
    elif choice == 's_li':
        @m.Constraint(pyo.RangeSet(T), m.period[1].fs.stages, m.period[1].fs.tubes)
        def li_sieving_con(b, t, i, j):
            """Set lithium deviations."""
            return m.period[t].fs.stage[i].sieving_coefficient['Li', j] == 1.3*m.dev[t]

        for t in pyo.RangeSet(T):
            for i in m.period[1].fs.stages:
                for j in m.period[1].fs.tubes:
                    m.period[t].fs.stage[i].sieving_coefficient['Li', j].unfix()
    elif choice == 's_co':
        @m.Constraint(pyo.RangeSet(T), m.period[1].fs.stages, m.period[1].fs.tubes)
        def co_sieving_con(b, t, i, j):
            """Set cobalt deviations."""
            return m.period[t].fs.stage[i].sieving_coefficient['Co', j] == 0.5*m.dev[t]
        
        for t in pyo.RangeSet(T):
            for i in m.period[1].fs.stages:
                for j in m.period[1].fs.tubes:
                    m.period[t].fs.stage[i].sieving_coefficient['Co', j].unfix()

def add_slacks(m):
    m.s = pyo.Var(['Li', 'Co'], bounds=(0,1))
    m.cost_objective.expr += 1e7*sum(m.s[i] for i in m.s)
    m.li_lb.set_value(m.li_lb.body + m.s['Li'] >= m.R)
    m.co_lb.set_value(m.co_lb.body + m.s['Co'] >= m.Rco)
    # diaf_slack_con = m.period[1].fs.diafiltrate_pump.costing.install_flows_constraint
    # diaf_slack_con.set_value(diaf_slack_con.body - m.s['pump'] <= 0)
    
    
choice = 'flow'                    
ramp = 0.1
add_multiperiod_unc_con(m, choice=choice)

solver = SolverFactory("gams:conopt")
result = solver.solve(m, tee=True)

uncparams = [m.dev]
box_set = pyros.BoxSet(bounds=[(0.5, 1.5) for _ in pyo.RangeSet(T)])

# polyhedral set
poly_ub = np.zeros((T,T))
np.fill_diagonal(poly_ub, 1)
np.fill_diagonal(poly_ub[1:, :-1], -1)
poly_lb = np.zeros((T,T))
np.fill_diagonal(poly_lb, -1)
np.fill_diagonal(poly_lb[1:, :-1], 1)
avg_dev = np.ones(T)
rlim = np.ones(2*T)*ramp
rlim[0] = 1 + ramp
rlim[T] =-(1 - ramp)
poly_set = pyros.PolyhedralSet(
    lhs_coefficients_mat=np.vstack((poly_ub, poly_lb, avg_dev, -avg_dev)),
    rhs_vec=np.append(rlim, [T, -T])
)
print(poly_set.coefficients_mat)
print(poly_set.rhs_vec)

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
fh = logging.FileHandler(f"disturbed_{choice}_ramplim_{ramp}.log")
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
    uncertainty_set=poly_set,
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

np.set_printoptions(threshold=np.inf)
print(np.round(alldrvars[:, :-1]))
print(np.round(alldrvars[:, -1]))
print([i[-1] for i in alldrvars_names])

# for t in pyo.RangeSet(T): utils.report_values(m.period[t])

