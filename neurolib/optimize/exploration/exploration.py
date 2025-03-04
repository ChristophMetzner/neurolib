import multiprocessing
import datetime
import os
import logging
import pathlib
import copy

import h5py
import pypet
import pandas as pd
import tqdm

from ...utils import paths
from ...utils import pypetUtils as pu

from ...utils.collections import dotdict


class BoxSearch:
    """
    Paremter box search for a given model and a range of parameters.
    """

    def __init__(
        self, model=None, parameterSpace=None, evalFunction=None, filename=None, saveAllModelOutputs=False,
    ):
        """Either a model has to be passed, or an evalFunction. If an evalFunction
        is passed, then the evalFunction will be called and the model is accessible to the 
        evalFunction via `self.getModelFromTraj(traj)`. The parameters of the current 
        run are accible via `self.getParametersFromTraj(traj)`.

        If no evaluation function is passed, then the model is simulated using `Model.run()`
        for every parameter.

        :param model: Model to run for each parameter (or model to pass to the evaluation funciton if an evaluation function is used), defaults to None
        :type model: `neurolib.models.model.Model`, optional
        :param parameterSpace: Parameter space to explore, defaults to None
        :type parameterSpace: `neurolib.utils.parameterSpace.ParameterSpace`, optional
        :param evalFunction: Evaluation function to call for each run., defaults to None
        :type evalFunction: function, optional
        :param filename: HDF5 storage file name, if left empty, defaults to ``exploration.hdf``
        :type filename: str
        :param saveAllModelOutputs: If True, save all outputs of model, else only default output of the model will be saved. Note: if saveAllModelOutputs==False and the model's parameter model.params['bold']==rue, then BOLD output will be saved as well, defaults to False
        :type saveAllModelOutputs: bool
        """
        self.model = model
        if evalFunction is None and model is not None:
            self.evalFunction = self.runModel
        elif evalFunction is not None:
            self.evalFunction = evalFunction

        assert (evalFunction is not None) or (
            model is not None
        ), "Either a model has to be specified or an evalFunction."

        assert parameterSpace is not None, "No parameters to explore."

        self.parameterSpace = parameterSpace
        self.exploreParameters = parameterSpace.dict()

        # todo: use random ICs for every explored point or rather reuse the ones that are generated at model initialization
        self.useRandomICs = False

        filename = filename or "exploration.hdf"
        self.filename = filename

        self.saveAllModelOutputs = saveAllModelOutputs

        # bool to check whether pypet was initialized properly
        self.initialized = False
        self.initializeExploration(self.filename)

    def initializeExploration(self, filename="exploration.hdf"):
        """Initialize the pypet environment
        
        :param filename: hdf filename to store the results in , defaults to "exploration.hdf"
        :type filename: str, optional
        """
        # create hdf file path if it does not exist yet
        pathlib.Path(paths.HDF_DIR).mkdir(parents=True, exist_ok=True)

        # set default hdf filename
        self.HDF_FILE = os.path.join(paths.HDF_DIR, filename)

        # initialize pypet environment
        trajectoryName = "results" + datetime.datetime.now().strftime("-%Y-%m-%d-%HH-%MM-%SS")
        trajectoryfilename = self.HDF_FILE

        nprocesses = multiprocessing.cpu_count()
        logging.info("Number of processes: {}".format(nprocesses))

        # set up the pypet environment
        env = pypet.Environment(
            trajectory=trajectoryName,
            filename=trajectoryfilename,
            multiproc=True,
            ncores=nprocesses,
            complevel=9,
            # log_stdout=False,
            # log_config=None,
            # report_progress=True,
            # log_multiproc=False,
        )
        self.env = env
        # Get the trajectory from the environment
        self.traj = env.trajectory
        self.trajectoryName = self.traj.v_name

        # Add all parameters to the pypet trajectory
        if self.model is not None:
            # if a model is specified, use the default parameter of the
            # model to initialize pypet
            self.addParametersToPypet(self.traj, self.model.params)
        else:
            # else, use a random parameter of the parameter space
            self.addParametersToPypet(self.traj, self.parameterSpace.getRandom(safe=True))

        # Tell pypet which parameters to explore
        self.pypetParametrization = pypet.cartesian_product(self.exploreParameters)
        logging.info(
            "Number of parameter configurations: {}".format(
                len(self.pypetParametrization[list(self.pypetParametrization.keys())[0]])
            )
        )

        self.traj.f_explore(self.pypetParametrization)

        # initialization done
        logging.info("BoxSearch: Environment initialized.")
        self.initialized = True

    def addParametersToPypet(self, traj, params):
        """This function registers the parameters of the model to Pypet.
        Parameters can be nested dictionaries. They are unpacked and stored recursively.
        
        :param traj: Pypet trajectory to store the parameters in
        :type traj: `pypet.trajectory.Trajectory`
        :param params: Parameter dictionary
        :type params: dict, dict[dict,]
        """

        def addParametersRecursively(traj, params, current_level):
            # make dummy list if just string
            if isinstance(current_level, str):
                current_level = [current_level]
            # iterate dict
            for key, value in params.items():
                # if another dict - recurse and increase level
                if isinstance(value, dict):
                    addParametersRecursively(traj, value, current_level + [key])
                else:
                    param_address = ".".join(current_level + [key])
                    traj.f_add_parameter(param_address, value)

        addParametersRecursively(traj, params, [])

    def saveOutputsToPypet(self, outputs, traj):
        """This function takes all outputs in the form of a nested dictionary
        and stores all data into the pypet hdf file.
        
        :param outputs: Simulation outputs as a dictionary.
        :type outputs: dict
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        """

        def makeSaveStringForPypet(value, savestr):
            """Builds the pypet-style results string from the results
            dictionary's keys.
            """
            for k, v in value.items():
                if isinstance(v, dict):
                    _savestr = savestr + k + "."
                    makeSaveStringForPypet(v, _savestr)
                else:
                    _savestr = savestr + k
                    self.traj.f_add_result(_savestr, v)

        assert isinstance(outputs, dict), "Outputs must be an instance of dict."
        value = outputs
        savestr = "results.$."
        makeSaveStringForPypet(value, savestr)

    def runModel(self, traj):
        """If not evaluation function is given, we assume that a model will be simulated.
        This function will be called by pypet directly and therefore wants a pypet trajectory as an argument
        
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        """
        if self.useRandomICs:
            logging.warn("Random initial conditions not implemented yet")
        # get parameters of this run from pypet trajectory
        runParams = traj.parameters.f_to_dict(short_names=True, fast_access=True)
        # set the parameters for the model
        self.model.params.update(runParams)

        # get kwargs from Exploration.run()
        runKwargs = {}
        if hasattr(self, "runKwargs"):
            runKwargs = self.runKwargs
        # run it
        self.model.run(**runKwargs)
        # save outputs
        self.saveModelOutputsToPypet(traj)

    def saveModelOutputsToPypet(self, traj):
        # save all data to the pypet trajectory
        if self.saveAllModelOutputs:
            # save all results from exploration
            self.saveOutputsToPypet(self.model.outputs, traj)
        else:
            # save only the default output
            self.saveOutputsToPypet({self.model.default_output: self.model.output, "t": self.model.outputs["t"]}, traj)
            # save BOLD output
            # if "bold" in self.model.params:
            #     if self.model.params["bold"] and "BOLD" in self.model.outputs:
            #         self.saveOutputsToPypet(self.model.outputs["BOLD"], traj)
            if "BOLD" in self.model.outputs:
                self.saveOutputsToPypet(self.model.outputs["BOLD"], traj)

    def getParametersFromTraj(self, traj):
        """Returns the parameters of the current run as a (dot.able) dictionary
        
        :param traj: Pypet trajectory
        :type traj: `pypet.trajectory.Trajectory`
        :return: Parameter set of the current run
        :rtype: dict
        """
        runParams = self.traj.parameters.f_to_dict(short_names=True, fast_access=True)
        return dotdict(runParams)

    def getModelFromTraj(self, traj):
        """Return the appropriate model with parameters for this individual
        :params traj: Pypet trajectory of current run

        :returns model: Model with the parameters of this individual.
        """
        model = self.model
        runParams = self.traj.parameters.f_to_dict(short_names=True, fast_access=True)
        model.params.update(runParams)
        return model

    def run(self, **kwargs):
        """
        Call this function to run the exploration
        """
        self.runKwargs = kwargs
        assert self.initialized, "Pypet environment not initialized yet."
        self.env.run(self.evalFunction)

    def loadResults(self, filename=None, trajectoryName=None, pypetShortNames=True):
        """Load results from a hdf file of a previous simulation.
        
        :param filename: hdf file name in which results are stored, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory inside the hdf file, newest will be used if left empty, defaults to None
        :type trajectoryName: str, optional
        :param pypetShortNames: Use pypet short names as keys for the results dictionary. Use if you are experiencing errors due to natural naming collisions.
        :type pypetShortNames: boolean
        """

        self.loadDfResults(filename, trajectoryName)

        # make a list of dictionaries with results
        logging.info("Creating results dictionary ...")
        self.results = []
        for rInd in tqdm.tqdm(range(self.nResults), total=self.nResults):
            self.pypetTrajectory.results[rInd].f_load()
            result = self.pypetTrajectory.results[rInd].f_to_dict(fast_access=True, short_names=pypetShortNames)
            result = dotdict(result)
            self.pypetTrajectory.results[rInd].f_remove()
            self.results.append(result)

        # Postprocess result keys if pypet short names aren't used
        # Before: results.run_00000001.outputs.rates_inh
        # After: outputs.rates_inh
        if pypetShortNames == False:
            for i, r in enumerate(self.results):
                new_dict = dotdict({})
                for key, value in r.items():
                    new_key = "".join(key.split(".", 2)[2:])
                    new_dict[new_key] = r[key]
                self.results[i] = copy.deepcopy(new_dict)

        logging.info("All results loaded.")

    def loadDfResults(self, filename=None, trajectoryName=None):
        """Load results from a hdf file of a previous simulation.
        
        :param filename: hdf file name in which results are stored, defaults to None
        :type filename: str, optional
        :param trajectoryName: Name of the trajectory inside the hdf file, newest will be used if left empty, defaults to None
        :type trajectoryName: str, optional
        """
        # chose HDF file to load
        if filename == None:
            filename = self.HDF_FILE
        self.pypetTrajectory = pu.loadPypetTrajectory(filename, trajectoryName)
        self.nResults = len(self.pypetTrajectory.f_get_run_names())

        exploredParameters = self.pypetTrajectory.f_get_explored_parameters()

        # create pandas dataframe of all runs with parameters as keys
        logging.info("Creating pandas dataframe ...")
        niceParKeys = [p.split(".")[-1] for p in exploredParameters.keys()]
        self.dfResults = pd.DataFrame(columns=niceParKeys, dtype=object)
        for nicep, p in zip(niceParKeys, exploredParameters.keys()):
            self.dfResults[nicep] = exploredParameters[p].f_get_range()

    def getRun(self, runId, filename=None, trajectoryName=None, pypetShortNames=True):
        """Load the simulated data of a run and its parameters from a pypetTrajectory.

        :param runId: ID of the run
        :type runId: int

        :return: Dictionary with simulated data and parameters of the run.
        :type return: dict        
        """
        # chose HDF file to load
        if filename == None:
            filename = self.HDF_FILE

        pypetTrajectory = None
        if hasattr(self, "pypetTrajectory"):
            pypetTrajectory = self.pypetTrajectory

        # if there was no pypetTrajectory loaded before
        if pypetTrajectory is None:
            # chose HDF file to load
            if filename == None:
                filename = self.HDF_FILE
            pypetTrajectory = pu.loadPypetTrajectory(filename, trajectoryName)

        return pu.getRun(runId, pypetTrajectory, pypetShortNames=pypetShortNames)
