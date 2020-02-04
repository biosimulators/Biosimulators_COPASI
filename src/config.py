""" Config for COPASI simulator

:Author: Akhil Teja  < akhilmteja@gmail.com >
:Date: 2020-01-28
:Copyright: 2019, UCONN Health
:License: MIT

"""

# -*- coding: ascii -*-
import os

if os.getenv('SIMULATION_ID') is None:    
    from dotenv import load_dotenv
    load_dotenv()


class Config:
    SEDML_BASE = '/usr/local/app/copasi/simulation'
    SIMULATION_ID = os.getenv('SIMULATION_ID')
    USER_SUB_ID = os.getenv('USER_SUB_ID')
    JOBHOOK_URL = os.getenv('JOBHOOK_URL')
    JOB_ID = os.getenv('JOB_ID')
    SEDML_DIR = os.path.join(SEDML_BASE, USER_SUB_ID, SIMULATION_ID)
    AUTH0_CLIENT_ID = os.getenv('AUTH0_CLIENT_ID')
    AUTH0_CLIENT_SECRET = os.getenv('AUTH0_CLIENT_SECRET')
    AUTH0_BIOSIMULATIONS_AUDIENCE = 'api.biosimulations.org'
    AUTH0_TOKEN_URL = 'https://crbm.auth0.com/oauth/token'
    AUTH0_GRANT_TYPE = 'client_credentials'
