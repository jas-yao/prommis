"""Multiple Cascade Case Study."""
import sys

from idaes.models.unit_models import Separator as Splitter
from pyomo.environ import (
    SolverFactory,
    TransformationFactory,
    assert_optimal_termination,
    value,
)
from pyomo.network import Arc
from prommis.nanofiltration.membrane_cascade_flowsheet.solute_property import (
    SoluteParameters,
)

from idaes.core.util import to_json, from_json
from idaes.core.util.model_diagnostics import DiagnosticsToolbox
from idaes.core.util.model_statistics import report_statistics

from prommis.nanofiltration.membrane_cascade_flowsheet.precipitator import Precipitator
from prommis.nanofiltration.membrane_cascade_flowsheet import utils
from prommis.nanofiltration.membrane_cascade_flowsheet.diafiltration_flowsheet_model import (
    DiafiltrationModel,
)
from prommis.nanofiltration.membrane_cascade_flowsheet.multistart_solve import MultistartSolve
from idaes.core import (
    FlowsheetBlock,
    MaterialBalanceType,
    MomentumBalanceType,
    UnitModelBlock,
    UnitModelCostingBlock,
)
from idaes.core.util.initialization import propagate_state
from idaes.core.util.scaling import constraint_autoscale_large_jac
from idaes.models.unit_models import (
    EnergySplittingType,
    Mixer,
    MixerInitializer,
    MixingType,
    MomentumMixingType,
    MSContactorInitializer,
)

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
solutes = ["Li", "Co", "Fe"]
flux = 0.1  # m3 / m2 / h
# for c1, use NF90: {"Li": 0.7, "Co": 0.3, "Fe": 0.03}
# for c2, use NF270: {"Li": 0.81, "Co": 0.15, "Fe": 0.17}
# sieving_coefficient = {"Li": 1.3, "Co": 0.5, "Fe": 0.1}
sieving_coefficient = {"Li": 0.7, "Co": 0.3, "Fe": 0.03}
feed = {
    "solvent": 100,  # m^3/hr of water
    "Li": 0.8 * 100,  # kg/hr
    "Co": 6.5 * 100,  # kg/hr
    "Fe": 0.5 * 100,  # kg/hr
}
diaf = {
    "solvent": 30,  # m^3/hr of water
    "Li": 0.1 * 30,  # kg/hr
    "Co": 0.2 * 30,  # kg/hr
    "Fe": 0.2 * 30,  # kg/hr
}
precipitate = False

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
        "permeate": {"Li": 0.81, "Co": 0.01, "Fe": 0.5},
        "retentate": {"Li": 0.20, "Co": 0.89, "Fe": 0.5},
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
cascade1 = df.build_cascade_block(mix_style, LiLB=0.5, CoLB=0.5, periods=1)

# cascade2 should be initialized with permeate outlet of cascade1
cascade2_feed = {
    'solvent': cascade1.period[1].fs.mix_product_permeate.mixed_state[0].flow_vol.value,
    'Li': cascade1.period[1].fs.mix_product_permeate.mixed_state[0].flow_mass_solute['Li'].value,
    'Co': cascade1.period[1].fs.mix_product_permeate.mixed_state[0].flow_mass_solute['Co'].value,
    'Fe': cascade1.period[1].fs.mix_product_permeate.mixed_state[0].flow_mass_solute['Fe'].value,
}
df.feed = cascade2_feed
df.sieving_coefficient = {"Li": 0.81, "Co": 0.15, "Fe": 0.17}
cascade2 = df.build_cascade_block(mix_style, LiLB=0.5, CoLB=0.5, periods=1)

# add each cascade to the same model
m = pyo.ConcreteModel()
m.fs = FlowsheetBlock(dynamic=False)
m.fs.properties = SoluteParameters(solutes=solutes)
m.fs.solutes = pyo.Set(initialize=solutes)
m.fs.c1 = cascade1
m.fs.c2 = cascade2

# connect cascade1 to splitter to cascade2
m.fs.split_c1 = Splitter(
    outlet_list=["recycle", "forward"],
    property_package=m.fs.properties,
    material_balance_type=MaterialBalanceType.componentTotal,
    momentum_balance_type=MomentumBalanceType.none,
    energy_split_basis=EnergySplittingType.none,
)
m.fs.cascade1_to_split_c1 = Arc(
    source=m.fs.c1.period[1].fs.mix_product_permeate.outlet,
    destination=m.fs.split_c1.inlet,
)
m.fs.split_c1_to_cascade2 = Arc(
    source=m.fs.split_c1.forward,
    destination=m.fs.c2.period[1].fs.split_feed.inlet,
)
# m.fs.cascade1_to_cascade2 = Arc(
#     source=m.fs.c1.period[1].fs.mix_product_permeate.outlet,
#     destination=m.fs.c2.period[1].fs.split_feed.inlet,
# )
m.fs.c2.period[1].fs.split_feed.inlet.unfix()
TransformationFactory("network.expand_arcs").apply_to(m)

# add precipitators
# 3 precipitators, one for each solute
m.fs.precipitator = Precipitator(
    ["Li", "Co", "Fe"],
    outlet_list=["solid", "downstream", "recycle"],
    yields={
        "Li": {"Li": 0.81, "Co": 0.99, "Fe": 0.99},
        "Co": {"Li": 0.20, "Co": 0.89, "Fe": 0.09},
        "Fe": {"Li": 0.01, "Co": 0.25, "Fe": 0.99},
    },
    property_package=m.fs.properties,
    material_balance_type=MaterialBalanceType.componentTotal,
    momentum_balance_type=MomentumBalanceType.none,
    energy_split_basis=EnergySplittingType.none,
)
m.fs.add_component(
    "connect_precipitator_Fe_inlet",
    Arc(
        source=m.fs.c1.period[1].fs.mix_product_retentate.outlet,
        destination=m.fs.precipitator["Fe"].inlet,
    ),
)
m.fs.add_component(
    "connect_precipitator_Co_inlet",
    Arc(
        source=m.fs.c2.period[1].fs.mix_product_retentate.outlet,
        destination=m.fs.precipitator["Co"].inlet,
    ),
)
m.fs.add_component(
    "connect_precipitator_Li_inlet",
    Arc(
        source=m.fs.c2.period[1].fs.mix_product_permeate.outlet,
        destination=m.fs.precipitator["Li"].inlet,
    ),
)

# connect recycles
# 3 splitters for each precipitator with 2 outlets to each cascade
# 2 recycle mixers into each train with 3 inlets from each precipitator
# m.fs.split_prec_recycles = Splitter(
#     ["Li", "Co", "Fe"],
#     outlet_list=["c1", "c2"],
#     property_package=m.fs.properties,
#     material_balance_type=MaterialBalanceType.componentTotal,
#     momentum_balance_type=MomentumBalanceType.none,
#     energy_split_basis=EnergySplittingType.none,
# )

# m.fs.prec_recycle_mixers = Mixer(
#     [1, 2],
#     inlet_list=["Li", "Co", "Fe"],
#     property_package=m.fs.properties,
#     material_balance_type=MaterialBalanceType.componentTotal,
#     energy_mixing_type=MixingType.none,
#     momentum_mixing_type=MomentumMixingType.none,
# )

# for prod in ["Li", "Co", "Fe"]:
#     m.fs.add_component(
#         f"connect_precipitator_{prod}_to_splitter",
#         Arc(
#             source=m.fs.precipitator[prod].recycle,
#             destination=m.fs.split_prec_recycles[prod].inlet,
#         ),
#     )
#     m.fs.add_component(
#         f"connect_precipitator_{prod}_split_to_cascade_1",
#         Arc(
#             source=m.fs.split_prec_recycles[prod].c1,
#             destination=getattr(m.fs.prec_recycle_mixers[1], f"{prod}"),
#         ),
#     )
#     m.fs.add_component(
#         f"connect_precipitator_{prod}_split_to_cascade_2",
#         Arc(
#             source=m.fs.split_prec_recycles[prod].c2,
#             destination=getattr(m.fs.prec_recycle_mixers[2], f"{prod}"),
#         ),
#     )
# for i in [1, 2]:
#     m.fs.add_component(
#         f"connect_prec_inlet_mixer_{i}",
#         Arc(
#             source=m.fs.prec_recycle_mixers[i].outlet,
#             destination=getattr(m.fs, f"c{i}").period[1].fs.split_diafiltrate.inlet,
#         ),
#     )

m.fs.prec_recycle_mixers = Mixer(
    [1, 2],
    inlet_list=["retentate", "permeate"],
    property_package=m.fs.properties,
    material_balance_type=MaterialBalanceType.componentTotal,
    energy_mixing_type=MixingType.none,
    momentum_mixing_type=MomentumMixingType.none,
)

m.fs.add_component(
    "connect_precipitator_Fe_to_rec_mixer",
    Arc(
        source=m.fs.precipitator["Fe"].recycle,
        destination=m.fs.prec_recycle_mixers[1].retentate,
    ),
)
m.fs.add_component(
    "connect_c1_permeate_to_rec_mixer",
    Arc(
        source=m.fs.split_c1.recycle,
        destination=m.fs.prec_recycle_mixers[1].permeate,
    ),
)
m.fs.add_component(
    "connect_precipitator_Co_to_rec_mixer",
    Arc(
        source=m.fs.precipitator["Co"].recycle,
        destination=m.fs.prec_recycle_mixers[2].retentate,
    ),
)
m.fs.add_component(
    "connect_precipitator_Li_to_rec_mixer",
    Arc(
        source=m.fs.precipitator["Li"].recycle,
        destination=m.fs.prec_recycle_mixers[2].permeate,
    ),
)
for i in [1, 2]:
    m.fs.add_component(
        f"connect_prec_inlet_mixer_{i}",
        Arc(
            source=m.fs.prec_recycle_mixers[i].outlet,
            destination=getattr(m.fs, f"c{i}").period[1].fs.split_diafiltrate.inlet,
        ),
    )
TransformationFactory("network.expand_arcs").apply_to(m)

m.fs.c1.period[1].fs.split_diafiltrate.inlet.unfix()
m.fs.c2.period[1].fs.split_diafiltrate.inlet.unfix()
for prod in ["Li", "Co", "Fe"]:
    m.fs.precipitator[prod].volume.unfix()
    # m.fs.precipitator[prod].yields["solvent", "recycle"].unfix()
    m.fs.precipitator[prod].split_inlet["bypass"].unfix()

    m.fs.precipitator[prod].yields["solvent", "recycle"].fix(0.6)
    for sol in df.solutes:
        m.fs.precipitator[prod].yields[sol, "recycle"].fix(0.005)

# add recovery lower bounds
m.fs.li_recovery = pyo.Param(initialize=0.7, mutable=True)
m.fs.co_recovery = pyo.Param(initialize=0.7, mutable=True)
m.fs.fe_recovery = pyo.Param(initialize=0.7, mutable=True)
m.fs.li_rec_lb = pyo.Constraint(expr=m.fs.precipitator["Li"].solid.flow_mass_solute[0, "Li"] >= m.fs.li_recovery*0.8 * 100)
m.fs.co_rec_lb = pyo.Constraint(expr=m.fs.precipitator["Co"].solid.flow_mass_solute[0, "Co"] >= m.fs.co_recovery*6.5 * 100)
m.fs.fe_rec_lb = pyo.Constraint(expr=m.fs.precipitator["Fe"].solid.flow_mass_solute[0, "Fe"] >= m.fs.fe_recovery*0.5 * 100)
# for prod in ["Li", "Co", "Fe"]:
#     m.fs.precipitator[prod].downstream_state[0].flow_vol.lb = 1

# new objective
m.overall_cost_objective = pyo.Objective(
    expr=cascade1.cost_objective.expr + cascade2.cost_objective.expr
)

solver = SolverFactory("gams:conopt")
result = solver.solve(m, tee=True)

print(result)
report_statistics(m)
vals = utils.report_values(m.fs.c1.period[1], prec=precipitate)
utils.visualize_flows(
    num_boxes=num_s, num_sub_boxes=num_t, num_sol=4, conf=mix_style, model=vals)
vals = utils.report_values(m.fs.c2.period[1], prec=precipitate)
utils.visualize_flows(
    num_boxes=num_s, num_sub_boxes=num_t, num_sol=4, conf=mix_style, model=vals
)
vals = utils.report_values(m.fs.c1.period[1], prec=precipitate)
fig, ax = utils.visualize_flows(
    num_boxes=num_s, num_sub_boxes=num_t, num_sol=4, conf=mix_style, model=vals, savefig=True
)
vals = utils.report_values(m.fs.c2.period[1], prec=precipitate)
fig, ax = utils.visualize_flows(
    num_boxes=num_s, num_sub_boxes=num_t, num_sol=4, conf=mix_style, model=vals, savefig=True
)

# TODO
# 1. Show initialization design (Co/Li)
# 2. Show optimized design (Co/Li)
# 3. Visualize flowsheet of combined cascades
# 4. Add in Fe for Fe/Co/Li separation (repeat analysis)
# 5. Consider case study with Li brine or semiconductor waste feedstock...
