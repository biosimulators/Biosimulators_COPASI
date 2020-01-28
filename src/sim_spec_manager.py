""" Simulation Spec Parser (from SEDML)

:Author: Akhil Teja  < akhilmteja@gmail.com >
:Date: 2020-01-28
:Copyright: 2019, UCONN Health
:License: MIT

"""

from COPASI import *
from .config import Config

class SimulationSpecManager:
    def __init__(self, job_id: str, jobhook_url):
        self.ALGORITHMS_MAP = {
            "0000089": CTaskEnum.Method_DsaLsodar,
            "0000304": CTaskEnum.Method_RADAU5,
            "0000027": CTaskEnum.Method_stochastic,
            "0000038": CTaskEnum.Method_directMethod,
            "0000039": CTaskEnum.Method_tauLeap,
            "0000333": CTaskEnum.Method_adaptiveSA,
            "0000064": CTaskEnum.Method_hybrid,
            "0000088": CTaskEnum.Method_hybridLSODA,
            "0000032": CTaskEnum.Method_hybridODE45,
            "0000318": CTaskEnum.Method_stochasticRunkeKuttaRI5
        }
        self.JOB_ID = job_id
        self.JOBHOOK_URL = jobhook_url
        self.ALGORITHM = None
        self.INITIAL_TIME = None
        self.NUMBER_OF_POINTS = None
        self.OUTPUT_START_TIME = None
        self.OUTPUT_END_TIME = None

    def parse_sim_config_from_sedml(self, sedml: str):
        # Use XML parser (from omex parser) to get timepoints, start time, etc.
        # Update those timepoints, etc in self variables
        pass

    def __get_sedml():
        path = Config.SEDML_DIR
        # Use path to scan for files with .sedml extension
        # Convert to string
        # pass to self.parse_sim_config_from_sedml
        pass
