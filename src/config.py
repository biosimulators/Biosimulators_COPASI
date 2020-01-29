""" Config for COPASI simulator

:Author: Akhil Teja  < akhilmteja@gmail.com >
:Date: 2020-01-28
:Copyright: 2019, UCONN Health
:License: MIT

"""

import os


class Config:
    SEDML_DIR = 'usr/local/app/copasi/'
    JOBHOOK_URL = os.getenv('JOBHOOK_URL')
    JOB_ID = os.getenv('JOB_ID')
    AUTH0_CLIENT_ID = os.getenv('AUTH0_CLIENT_ID')
    AUTH0_CLIENT_SECRET = os.getenv('AUTH0_CLIENT_SECRET')
    AUTH0_BIOSIMULATIONS_AUDIENCE = 'api.biosimulations.org'
    AUTH0_TOKEN_URL = 'https://crbm.auth0.com/oauth/token'
    AUTH0_GRANT_TYPE = 'client_credentials'

