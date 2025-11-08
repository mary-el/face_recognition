import json
import logging
from datetime import datetime
from enum import Enum
from typing import Dict, Optional

import pandas as pd
import requests
from requests import RequestException


class DoorState(Enum):
    CLOSED = 0
    EXIT = 1
    ENTRANCE = 2


logger: Optional[logging.Logger] = None
connection: Optional["Connection"] = None


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

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        url = f'http://{self.host}{path}'
        try:
            response = requests.request(method, url, headers=self.headers, timeout=5, **kwargs)
            response.raise_for_status()
        except RequestException as exc:
            if logger:
                logger.exception("Request to %s failed: %s", url, exc)
            raise RuntimeError(f"Connection error while calling {url}") from exc
        return response

    def getToken(self) -> str:
        payload = {
            "login": self.login,
            "password": self.password
        }
        response = self._request("post", "/api/system/auth", json=payload)
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid auth response payload") from exc

        token = data.get("token")
        if not token:
            raise RuntimeError("Auth response missing token")
        return token

    def set_headers(self):
        self.headers["Authorization"] = f"Bearer {self.getToken()}"

    def read_users(self) -> Dict[int, str]:
        url = "/api/users/staff/list"
        querystring = {"withPhone": "true"}
        response = self._request("get", url, params=querystring)
        try:
            users_raw = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid users response payload") from exc

        db: Dict[int, str] = {}
        for user in users_raw:
            try:
                db[user['id']] = user['name']
            except KeyError:
                if logger:
                    logger.warning("Skipping malformed user record: %s", user)
        return db

    def passing(self, user_id, direction, event_description) -> bool:
        payload = {
            "user_id": user_id,
            "direction": direction,
            "event_description": event_description
        }
        response = self._request(
            "post",
            f'/api/devices/{self.config["turnstiles"]["id_tur"]}/pass',
            json=payload
        )
        try:
            data = response.json()
        except json.JSONDecodeError as exc:
            raise RuntimeError("Invalid pass response payload") from exc
        return data.get('result') == 'ok'

    def open_doors(self, id, direction: DoorState, user_name: str):
        if logger:
            logger.info('%s door opened for %s', self.directions[direction], user_name)
        time_diff = (datetime.now() - self.previous_state['time']).total_seconds()

        if id != self.previous_state['id'] or time_diff > self.config["turnstiles"]["min_time_diff"]:
            self.previous_state['id'] = id
            self.previous_state['direction'] = direction
            self.previous_state['time'] = datetime.now()
            try:
                success = self.passing(int(id), direction.value, f'{direction.name} door')
            except RuntimeError as exc:
                if logger:
                    logger.error("Failed to notify door opening: %s", exc)
                return

            if not success:
                if logger:
                    logger.warning('Authorization token expired, refreshing and retrying')
                try:
                    self.set_headers()
                    success = self.passing(int(id), direction.value, f'{direction.name} door')
                except RuntimeError as exc:
                    if logger:
                        logger.error("Retry failed to notify door opening: %s", exc)
                    return
                if not success and logger:
                    logger.error('Door open request rejected after token refresh')


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


def load_users(config):
    if config['source'] == 'excel':
        users = read_excel(config['excel_file'])
    else:
        conn = get_connection(config)
        users = conn.read_users()
    users[0] = config['no_name_user']
    return users
