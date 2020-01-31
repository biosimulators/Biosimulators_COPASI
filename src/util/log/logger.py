""" Logger for COPASI simulator

:Author: Akhil Teja  < akhilmteja@gmail.com >
:Date: 2020-01-29
:Copyright: 2019, UCONN Health
:License: MIT

"""
from config import Config
import requests
import json


# TODO: Implement proper levels of logging

# TODO: remove redundant code in different log methods
class Logger:
    def __init__(self,
                 push_to_crbmapi=False,
                 client_id=Config.AUTH0_CLIENT_ID,
                 client_secret=Config.AUTH0_CLIENT_SECRET,
                 client_audience=Config.AUTH0_BIOSIMULATIONS_AUDIENCE,
                 auth_url=Config.AUTH0_TOKEN_URL,
                 grant_type=Config.AUTH0_GRANT_TYPE,
                 jobhook_url=Config.JOBHOOK_URL,
                 sim_id=Config.SIMULATION_ID,
                 job_id=Config.JOB_ID
                 ):
        request_payload = 'grant_type={}&client_id={}&client_secret={}&audience={}'.format(
            grant_type, client_id, client_secret, client_audience
        )
        request_headers = {
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        self.jobhook_url = jobhook_url
        self.access_token = self.__get_access_token__(auth_url, request_headers, request_payload)
        self.jobhook_headers = {
            'Authorization': 'Bearer {}'.format(self.access_token)
        }
        self.push_to_crbmapi = push_to_crbmapi
        self.simulation_id = sim_id
        self.job_id = job_id

    def info(self, message: str):
        self.__log__('I', message)

    def warning(self, message: str):
        self.__log__('W', message)

    def error(self, message: str):
        self.__log__('E', message)

    def __log__(self, log_type: str, message):
        message = '{}: {}'.format(log_type, message)
        if self.push_to_crbmapi:
            requests.post(self.jobhook_url,
                          headers=self.jobhook_headers,
                          data=json.dumps({
                              'simId': self.simulation_id,
                              'jobId': self.job_id,
                              'message': message
                          })
                          )

    def __get_access_token__(self, url: str, request_headers: dict, request_payload: str):
        response = requests.post(url, headers=request_headers, data=request_payload)
        return response.text.encode('utf8')['access_token']
