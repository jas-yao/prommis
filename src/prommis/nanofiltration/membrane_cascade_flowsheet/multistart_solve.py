"""A multistart solver for better locally optimal solutions."""

from typing import Tuple
from pyomo.common.config import (
    ConfigDict,
)
from pyomo.opt import SolverFactory, SolverStatus
from pyomo.opt import TerminationCondition
from pyomo.environ import value, Var, Objective
from idaes.core.util.model_statistics import degrees_of_freedom
import numpy as np
import multiprocessing as mp

_author_ = "Jason L Yao"


def load_sol(model, sol):
    """
    Load solution (dict if variable names and values)
    to model.
    """
    for key, val in sol.items():
        model.find_component(key).set_value(val)
    return model

@SolverFactory.register('multistart_solve',
                        doc='Customized solver for finding better locally optimal solutions.')
class MultistartSolve(object):
    """
    Custom solver for debugging.

    The outline for this class was provided by John Siirola in
    https://stackoverflow.com/questions/74179391/custom-python-solver-for-pyomo

    This class wraps any provided exisiting solver and adds
    additional exporting utility to the `solve` function for debugging.
    """

    _dsolver = None
    _subsolver = None
    _label = None
    _mix = None
    _multiperiod = False
    _max_sense = False

    @property
    def label(self):
        """Get the solver."""
        return self._label

    @label.setter
    def label(self, name):
        """Set the solver."""
        self._label = name

    @property
    def mix(self):
        """Get the superstructure config."""
        return self._mix

    @mix.setter
    def mix(self, name):
        """Set the superstructure config."""
        self._mix = name

    @property
    def multiperiod(self):
        """Get the multiperiod setting."""
        return self._multiperiod

    @multiperiod.setter
    def multiperiod(self, val):
        """Set the multiperiod config."""
        self._multiperiod = val 

    @property
    def max_sense(self):
        """Get the minmax setting."""
        return self._max_sense

    @max_sense.setter
    def max_sense(self, val):
        """Set the minmax config."""
        self._max_sense = val 


    @property
    def dsolver(self):
        """Get the solver."""
        return self._dsolver

    @dsolver.setter
    def dsolver(self, opt):
        """Set the solver."""
        self._dsolver = opt
        if type(opt) is str:
            self._subsolver = SolverFactory(self._dsolver)
        else:
            self._subsolver = opt

    def solve(self, model, **kwds):
        """Solve method for debugging use."""
        saved_model = model
        
        # set the solver
        solver = self._subsolver

        # NOTE could have preprocessing/initialization/etc. steps here
        presolver = SolverFactory('gams:conopt')
        presolver.options["add_options"] = [
            "GAMS_MODEL.optfile = 1;",
            "$onecho > conopt.opt", 
            # Add solver options here:
            "ResLim = 1;",
            "$offecho",
        ]
        # NS = len(model.fs.stages)
        # NT = len(model.fs.tubes)
        # mix = self._mix
        solver_status = {SolverStatus.ok}
        termination_condition = {TerminationCondition.optimal,
                                TerminationCondition.locallyOptimal}

        # flows to fix for different initial points
        if self._multiperiod:
            init_flows = [
                # feed
                "period[1].fs.split_feed.split_fraction[0, outlet_1]",
                "period[1].fs.split_feed.split_fraction[0, outlet_2]",
                "period[1].fs.split_feed.split_fraction[0, outlet_3]",
                # diafiltrate (not used in many cases)
                # "period[1].fs.split_diafiltrate.split_fraction[0, outlet_1]",
                # "period[1].fs.split_diafiltrate.split_fraction[0, outlet_2]",
                # "period[1].fs.split_diafiltrate.split_fraction[0, outlet_3]",
                # permeate
                "period[1].fs.split_permeate[1].split_fraction[0, product]",
                "period[1].fs.split_permeate[2].split_fraction[0, product]",
                "period[1].fs.split_permeate[1].split_fraction[0, forward]",
                "period[1].fs.split_permeate[2].split_fraction[0, forward]",
                # retentate
                "period[1].fs.split_retentate[2].split_fraction[0, product]",
                "period[1].fs.split_retentate[3].split_fraction[0, product]",
                "period[1].fs.split_retentate[2].split_fraction[0, recycle]",
                "period[1].fs.split_retentate[3].split_fraction[0, recycle]",
            ]
        else:
            init_flows = [
                # feed
                "fs.split_feed.split_fraction[0, outlet_1]",
                "fs.split_feed.split_fraction[0, outlet_2]",
                "fs.split_feed.split_fraction[0, outlet_3]",
                # diafiltrate (not used in many cases)
                # "fs.split_diafiltrate.split_fraction[0, outlet_1]",
                # "fs.split_diafiltrate.split_fraction[0, outlet_2]",
                # "fs.split_diafiltrate.split_fraction[0, outlet_3]",
                # permeate
                "fs.split_permeate[1].split_fraction[0, product]",
                "fs.split_permeate[2].split_fraction[0, product]",
                "fs.split_permeate[1].split_fraction[0, forward]",
                "fs.split_permeate[2].split_fraction[0, forward]",
                # retentate
                "fs.split_retentate[2].split_fraction[0, product]",
                "fs.split_retentate[3].split_fraction[0, product]",
                "fs.split_retentate[2].split_fraction[0, recycle]",
                "fs.split_retentate[3].split_fraction[0, recycle]",
            ]

        objs = {}
        solns = {}
        all_res = {}
        for i in init_flows:
            # TODO: Maybe should initialize each time using the same starting point rather than using the previous for each iteration
            #       This could help keep everything from failing after the first iteration fails
            # TODO: Could parallelize solving each starting point
            model = saved_model.clone()
            print(f'Fixing flow for {i}')
            model.find_component(i).fix(0.95)
            try:
                res = presolver.solve(model, tee=False)
                if (res.solver.status in solver_status) and (
                        res.solver.termination_condition in termination_condition):
                    print(res.solver.status)
                    print(res.solver.termination_condition)
                    # mod_res = utils.report_values(m)
                    # utils.visualize_flows(NS, NT, conf=mix, model=mod_res, savefig=True, fname=f'./multistart{i}_start.png')
                else:
                    print('Failed to find optimal solution during presolve. Continuing to next step.')
            except:
                print('Failed to find optimal solution during presolve. Continuing to next step.')
            model.find_component(i).unfix()
            try:
                res = presolver.solve(model, tee=False)
                if (res.solver.status in solver_status) and (
                        res.solver.termination_condition in termination_condition):
                    print(res.solver.status)
                    print(res.solver.termination_condition)
                    # utils.report_cost(m)
                    # mod_res = utils.report_values(m)
                    # utils.visualize_flows(NS, NT, conf=mix, model=mod_res, savefig=True, fname=f'./multistart{i}_final.png')
                    model_objectives = [obj for obj in model.component_data_objects(Objective, active=True)]
                else:
                    print('Failed to find optimal solution during presolve. Continuing to next step.')
                    model_objectives = [1e12]
            except:
                print('Failed to find optimal solution during presolve. Continuing to next step.')
                model_objectives = [1e12]


            # make sure only one objective is active
            assert len(model_objectives) == 1
            objs[i] = value(model_objectives[0])
            solns[i] = {
                var.name: value(var)
                for var in model.component_data_objects(Var)
            }
            all_res[i] = res
        max_idx = np.argmax(list(objs.values()))
        min_idx = np.argmin(list(objs.values()))

        # final solve with best point
        print(objs)
        model = saved_model
        if self._max_sense:
            load_sol(model, list(solns.values())[max_idx])
            results = list(all_res.values())[max_idx]
        else:
            load_sol(model, list(solns.values())[min_idx])
            results = list(all_res.values())[min_idx]

        # for i in range(NS):
        #     # TODO: Maybe should initialize each time using the same starting point rather than using the previous for each iteration
        #     #       This could help keep everything from failing after the first iteration fails
        #     # TODO: Could parallelize solving each starting point
        #     model = saved_model.clone()
        #     print(f'Stage: {i+1}')
        #     if mix == 'tube':
        #         loc = i*NT + 1
        #     else:
        #         loc = i + 1
        #     model.fs.split_feed.split_fraction[0, f'outlet_{loc}'].fix(0.95)
        #     try:
        #         res = presolver.solve(model, tee=False)
        #         if (res.solver.status in solver_status) and (
        #                 res.solver.termination_condition in termination_condition):
        #             print(res.solver.status)
        #             print(res.solver.termination_condition)
        #             # mod_res = utils.report_values(m)
        #             # utils.visualize_flows(NS, NT, conf=mix, model=mod_res, savefig=True, fname=f'./multistart{i}_start.png')
        #         else:
        #             print('Failed to find optimal solution during presolve. Continuing to next step.')
        #     except:
        #         print('Failed to find optimal solution during presolve. Continuing to next step.')
        #     model.fs.split_feed.split_fraction[0, f'outlet_{loc}'].unfix()
        #     try:
        #         res = presolver.solve(model, tee=False)
        #         if (res.solver.status in solver_status) and (
        #                 res.solver.termination_condition in termination_condition):
        #             print(res.solver.status)
        #             print(res.solver.termination_condition)
        #             # utils.report_cost(m)
        #             # mod_res = utils.report_values(m)
        #             # utils.visualize_flows(NS, NT, conf=mix, model=mod_res, savefig=True, fname=f'./multistart{i}_final.png')
        #             model_objectives = [obj for obj in model.component_data_objects(Objective, active=True)]
        #         else:
        #             print('Failed to find optimal solution during presolve. Continuing to next step.')
        #             model_objectives = [1e12]
        #     except:
        #         print('Failed to find optimal solution during presolve. Continuing to next step.')
        #         model_objectives = [1e12]


        #     # make sure only one objective is active
        #     assert len(model_objectives) == 1
        #     objs[i] = value(model_objectives[0])
        #     solns[i] = {
        #         var.name: value(var)
        #         for var in model.component_data_objects(Var)
        #     }
        # max_idx = np.argmax(list(objs.values()))
        # min_idx = np.argmin(list(objs.values()))

        # # final solve with best point
        # model = saved_model
        # load_sol(model, solns[min_idx])

        # results = solver.solve(model, **kwds)
        
        return results


    def check_subsolver(self):
        """Check if subsolver exists."""
        if self._subsolver is None:
            raise TypeError('Solver has not been specified.')

    def available(self, exception_flag=True) -> bool:
        """Check if subsolver is available."""
        self.check_subsolver()
        return self._subsolver.available(exception_flag=exception_flag)

    def license_is_valid(self) -> bool:
        """Check if subsolver license is valid."""
        self.check_subsolver()
        return self._subsolver.license_is_valid()

    def version(self) -> Tuple:
        """Debug_Solve is just a draft."""
        return (0, 0, 1)

    @property
    def options(self) -> ConfigDict:
        """Pass this to the subsolver."""
        return self._subsolver.options

    @options.setter
    def options(self, val):
        """Pass this to the subsolver."""
        self._subsolver.options = val
