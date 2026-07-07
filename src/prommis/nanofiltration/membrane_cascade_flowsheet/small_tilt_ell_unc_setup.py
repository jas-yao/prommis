from matplotlib import pyplot as plt
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition, SolverResults
from idaes.core.util.model_statistics import report_statistics
from prommis.nanofiltration.membrane_cascade_flowsheet import utils
import pyomo.contrib.pyros as pyros
from prommis.nanofiltration.membrane_cascade_flowsheet.pyros_cost_settings import pyros_settings
import logging
import numpy as np

def unc_setup(m, p, k, c, loc, indiv=True):
    """
    Set up PyROS.

    m: Pyomo model
    p: perturbed %
    k: tilt parameter
    c: scaling
    loc: location of parameter
         e.g. {'stage': [1, 2], 'tube': [3, 4]}
    """
    # uncertain param locations
    stages = loc['stage']
    tubes = loc['tube']

    # make sure fbbt doesn't mess things up
    for i in m.fs.stages:
        m.fs.stage[i].sieving_coefficient.setlb(None)
        m.fs.stage[i].sieving_coefficient.setub(None)
                

    if indiv:
        if not hasattr(m, 'S_Co'):
            # Nominal parameters
            m.S_Co = pyo.Param(
                stages,
                tubes,
                initialize=pyo.value(m.fs.stage[1].sieving_coefficient['Co', 1]),
                mutable=True
            )
            m.S_Li = pyo.Param(
                stages,
                tubes,
                initialize=pyo.value(m.fs.stage[1].sieving_coefficient['Li', 1]),
                mutable=True
            )

            # constraint connecting S_Co/S_Li to all sieving coefficients
            @m.Constraint(stages, tubes)
            def S_Co_con(b, i, j):
                """Set all Co sieving equal to uncertain S."""
                return b.fs.stage[i].sieving_coefficient['Co', j] == b.S_Co[i, j]

            @m.Constraint(stages, tubes)
            def S_Li_con(b, i, j):
                """Set all Li sieving equal to uncertain S."""
                return b.fs.stage[i].sieving_coefficient['Li', j] == b.S_Li[i, j]

            for i in stages:
                for j in tubes:
                    m.fs.stage[i].sieving_coefficient['Co', j].unfix()
                    m.fs.stage[i].sieving_coefficient['Li', j].unfix()

        # uncertain parameters
        # unc_params_Co = [m.S_Co]
        # unc_params_Li = [m.S_Li]
        unc_params = []
        for i in stages:
            for j in tubes:
                unc_params.append(m.S_Li[i, j])
                unc_params.append(m.S_Co[i, j])
        means = [i.value for i in unc_params]
        print(means)

        # tilted ellipsoid set
        cov_mat = construct_cov_mat(m, p, k, c, stages, tubes)
        print(cov_mat)
        tilt_ell = pyros.EllipsoidalSet(
            center=means,
            shape_matrix=cov_mat,
            scale=c**2,
        )
    else:
        if not hasattr(m, 'S_Co'):
            # Nominal parameters
            m.S_Co = pyo.Param(
                initialize=pyo.value(m.fs.stage[1].sieving_coefficient['Co', 1]),
                mutable=True
            )
            m.S_Li = pyo.Param(
                initialize=pyo.value(m.fs.stage[1].sieving_coefficient['Li', 1]),
                mutable=True
            )

            # constraint connecting S_Co/S_Li to all sieving coefficients
            @m.Constraint(stages, tubes)
            def S_Co_con(b, i, j):
                """Set all Co sieving equal to uncertain S."""
                return b.fs.stage[i].sieving_coefficient['Co', j] == b.S_Co

            @m.Constraint(stages, tubes)
            def S_Li_con(b, i, j):
                """Set all Li sieving equal to uncertain S."""
                return b.fs.stage[i].sieving_coefficient['Li', j] == b.S_Li

            for i in stages:
                for j in tubes:
                    m.fs.stage[i].sieving_coefficient['Co', j].unfix()
                    m.fs.stage[i].sieving_coefficient['Li', j].unfix()

        unc_params = [m.S_Li, m.S_Co]
        means = [i.value for i in unc_params]
        print(means)

        # tilted ellipsoid set
        cov_mat = construct_cov_mat(m, p, k, c, [1], [1])
        print(cov_mat)
        tilt_ell = pyros.EllipsoidalSet(
            center=means,
            shape_matrix=cov_mat,
            scale=c**2,
        )

    return tilt_ell, unc_params


def construct_cov_mat(m, p, k, c, stages, tubes):
    """
    Construct covariance matrix for membrane manufacturing variability.
    """
    import math
    # parameterization of the ellipsoidal uncertainty set
    # NS = len(stages)
    # NT = len(tubes)
    # means = [m.fs.stage[1].sieving_coefficient["Li", 1].value, m.fs.stage[1].sieving_coefficient["Co", 1].value]
    # cov_mat = np.zeros((2*NS*NT, 2*NS*NT))
    # theta = (math.pi / 4)*k
    # for i in range(int((NS*NT))):
    #     cov_mat[2*i, 2*i] = (p*means[0]/3)**2 * math.cos(theta)**2 + (p*means[1]/3)**2 * math.sin(theta)**2
    #     cov_mat[2*i, 2*i+1] = (p*means[0]/3)**2 * math.cos(theta)*math.sin(theta) - (p*means[1]/3)**2 * math.sin(theta)*math.cos(theta)
    #     cov_mat[2*i+1, 2*i] = (p*means[0]/3)**2 * math.sin(theta)*math.cos(theta) - (p*means[1]/3)**2 * math.cos(theta)*math.sin(theta)
    #     cov_mat[2*i+1, 2*i+1] = (p*means[0]/3)**2 * math.sin(theta)**2 + (p*means[1]/3)**2 * math.cos(theta)
    NS = 1
    NT = 1
    means = [m.fs.stage[1].sieving_coefficient["Li", 1].value, m.fs.stage[1].sieving_coefficient["Co", 1].value]
    sigma_x = (means[0]*(1 + p) - means[0])/3
    sigma_y = (means[1]*(1 + p) - means[1])/3
    rho = k

    cov_mat = np.zeros((2*NS*NT, 2*NS*NT))

    cov_mat[0, 0] = sigma_x**2
    cov_mat[0, 1] = rho*sigma_x*sigma_y
    cov_mat[1, 0] = rho*sigma_x*sigma_y
    cov_mat[1, 1] = sigma_y**2

    return cov_mat

def flux_setup(m, p, u, loc):
    """
    Set up PyROS.

    m: Pyomo model
    p: perturbed %
    u: underperforming tubes
    loc: location of parameter
         e.g. {'stage': [1, 2], 'tube': [3, 4]}
    """
    # uncertain param locations
    stages = loc['stage']
    tubes = loc['tube']
    flux = pyo.value(m.fs.stage[1].flux[1])
    dev = p
    und_tubes = u
    assert und_tubes <= len(stages)*len(tubes)

    if not hasattr(m, 'J'):
        # Nominal parameters
        m.J = pyo.Param(
            stages,
            tubes,
            initialize=pyo.value(flux),
            mutable=True
        )

        # constraint connecting S_Co/S_Li to all sieving coefficients
        @m.Constraint(stages, tubes)
        def J_con(b, i, j):
            """Set all Co sieving equal to uncertain S."""
            return b.fs.stage[i].flux[j] == b.J[i, j]

        for i in stages:
            for j in tubes:
                m.fs.stage[i].flux[j].unfix()

    # uncertain parameters
    # unc_params_Co = [m.S_Co]
    # unc_params_Li = [m.S_Li]
    unc_params = []
    for i in stages:
        for j in tubes:
            unc_params.append(m.J[i, j])

    scenarios = []
    tot_tubes = len(stages)*len(tubes)

    from itertools import combinations

    # found at
    # https://stackoverflow.com/questions/43816965/permutation-without-duplicates-in-python
    def generate_comb(size, count):
        for positions in combinations(range(size), count):
            p = [flux] * size
            for i in positions:
                p[i] = flux - dev*flux
            yield p

    for i in pyo.RangeSet(und_tubes):
        combs = generate_comb(tot_tubes, i)
        for scene in combs:
            scenarios.append(scene)

    temp = []
    for i in stages:
        for j in tubes:
            temp.append(pyo.value(m.fs.stage[i].flux[j]))
    scenarios.append(temp)
    for i in scenarios:
        print(i)
    uncset = pyros.DiscreteScenarioSet(scenarios=scenarios)

    return uncset, unc_params
