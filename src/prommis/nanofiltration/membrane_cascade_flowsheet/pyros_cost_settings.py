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
    first, second = utils.sep_dof(m, mixing, first, second)
    print('Number of Second Stage Vars:', len(second))

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
