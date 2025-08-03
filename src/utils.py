import json
import logging
from datetime import datetime
from enum import Enum

import cv2
import pandas as pd
import requests


class DoorState(Enum):
    CLOSED = 0
    EXIT = 1
    ENTRANCE = 2


logger: logging.Logger = None
connection = None

def setup_logger(config) -> logging.Logger:
    global logger
    if logger:
        return logger
    logger = logging.getLogger(__name__)
    log_file = config["log_file"]

    if not logger.hasHandlers():
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s"
        )

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class Connection:
    def __init__(self, config):
        self.config = config
        self.connection_config = config['connection']
        self.host = self.connection_config["host"]
        self.login = self.connection_config["login"]
        self.password = self.connection_config["password"]

        self.headers = {
            "Content-type": "application/json; charset=UTF-8",
            "Authorization": "Bearer null"
        }
        self.previous_state = {
            'id': -1,
            'direction': 0,
            'time': datetime.now()
        }
        self.directions = {
            DoorState.ENTRANCE: "Entrance",
            DoorState.EXIT: "Exit"
        }

    def getToken(self):  # autorization
        url = f'http://{self.host}/api/system/auth'
        payload = {
            "login": self.login,
            "password": self.password
        }
        return requests.request("post", url, json=payload, headers=self.headers).json()["token"]

    def set_headers(self):
        self.headers["Authorization"] = self.getToken()

    def read_users(self):  # get list of users
        db = {}
        url = f'http://{self.host}/api/users/staff/list'
        querystring = {"withPhone": "true"}
        for user in json.loads(requests.request("get", url, headers=self.headers, params=querystring).text):
            db[user['id']] = user['name']
        return db

    def passing(self, user_id, direction, event_description):
        url = f'http://{self.host}/api/devices/{self.config["turnstiles"]["id_tur"]}/pass'
        payload = {
            "user_id": user_id,
            "direction": direction,
            "event_description": event_description
        }
        response = requests.request("post", url, json=payload, headers=self.headers)
        return (json.loads(response.text).get('result'))

    def open_doors(self, id, direction: DoorState, user_name):
        logger.info(f'{self.directions[direction]} door opened for {user_name}')
        time_diff = (datetime.now() - self.previous_state['time']).total_seconds()

        if id != self.previous_state['id'] or time_diff > self.config["turnstiles"]["min_time_diff"]:
            self.previous_state['id'] = id
            self.previous_state['direction'] = direction
            self.previous_state['time'] = datetime.now()
            if not (self.passing(int(id), direction, f'{direction} door') == 'ok'):  # open the door
                logger.error(f'Authorization token is out of date')
                exit()


def get_connection(config) -> Connection:
    global connection
    if connection:
        return connection
    connection = Connection(config)
    return connection


def read_excel(file):
    df = pd.read_excel(file)
    user_dict = {ind: name for ind, name in df.values}
    return user_dict
