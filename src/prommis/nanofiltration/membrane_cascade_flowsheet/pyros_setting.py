"""Setup for PyROS settings."""
import pyomo.environ as pyo
from prommis.nanofiltration.membrane_cascade_flowsheet import utils


def pyros_settings(m, mix):
    """Set PyROS solvers; x, z variables, priority order."""
    pyros_solver = pyo.SolverFactory("pyros")
    # solvers
    local_solver = pyo.SolverFactory('gams:conopt')
    global_solver = pyo.SolverFactory('gams:baron')
    global_solver.options["add_options"] = [
        "option reslim=600;",
        "GAMS_MODEL.optfile = 1;",
        "$onecho > baron.opt",
        "LPSol 3",
        "NLPSol 6",
        "$offecho",
    ]

    # designate variables
    mixing = mix
    first = []
    # for t in m.period:
    for t in [1]:
        first.extend(
            [
                m.period[t].fs.stage[1].length,
                m.period[t].fs.diafiltrate_pump.costing.install_inlet_vol_flow,
                m.period[t].fs.feed_pump.costing.install_inlet_vol_flow,
                m.period[t].fs.precipitator["retentate"].volume,
                m.period[t].fs.precipitator["permeate"].volume,
                m.period[t].fs.precipitator["retentate"].yields["solvent", "recycle"],
                m.period[t].fs.precipitator["permeate"].yields["solvent", "recycle"],
            ]
        )
        
    second = []
    second = []
    for t in m.period:
        second_T = []
        second_T.extend(
            [
                m.period[t].fs.split_diafiltrate.mixed_state[0].flow_vol,
                m.period[t].fs.precipitator["retentate"].split_inlet["bypass"],
                m.period[t].fs.precipitator["permeate"].split_inlet["bypass"],
            ]
        )
        first, second_T = utils.sep_dof(m.period[t], mixing, first, second_T)
        second.append(second_T)

    print('Number of Second Stage Vars:', sum(len(second[t-1]) for t in m.period))

    porder = {}
    # for idx, i in enumerate(second):
    #     lower = pyo.value(i.lb)
    #     upper = pyo.value(i.ub)

    #     i.setlb(None)
    #     i.setub(None)
    #     # pname = i.name

    #     if lower is not None:
    #         m.add_component(f"ss_var_{idx}_lb", pyo.Constraint(expr=i >= lower))
    #         porder[f'ss_var_{idx}_lb'] = 1
    #     if upper is not None:
    #         m.add_component(f"ss_var_{idx}_ub", pyo.Constraint(expr=i <= upper))
    #         porder[f'ss_var_{idx}_ub'] = 1
    porder['prec_li_lb'] = 2
    porder['prec_co_lb'] = 2
    # porder['epigraph_constr'] = 1
    return pyros_solver, local_solver, global_solver, first, second, porder
