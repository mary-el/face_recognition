import json
import pickle
from datetime import datetime

import pandas as pd
import requests

from config import config

connection_config = config['connection']
log_file = config['log_file']
headers = {
    "Content-type": "application/json; charset=UTF-8",
    "Authorization": "Bearer null"
}

previous_state = {
    'id': -1,
    'direction': 0
}


def set_headers():
    global headers
    headers["Authorization"] = getToken(connection_config['host'])


def getToken(host):  # авторизация
    url = f'http://{host}/api/system/auth'
    payload = {
        "login": connection_config['login'],
        "password": connection_config['password']
    }
    return requests.request("post", url, json=payload, headers=headers).json()["token"]


def read_users():  # Получение списка сотрудников
    url = f'http://{connection_config["host"]}/api/users/staff/list'
    querystring = {"withPhone": "true"}
    response = pd.read_json(requests.request("get", url, headers=headers, params=querystring))[['id', 'name']]
    return response.text


def passing(user_id, direction, event_description='Камера'):
    '''
    direction 1 на вход 2 на выход  направление  открытие турникета
    '''
    url = f'http://{connection_config["host"]}/api/devices/{config["tourniquets"]["id_tur"]}/pass'
    payload = {
        "user_id": user_id,
        "direction": direction,
        "event_description": event_description
    }
    response = requests.request("post", url, json=payload, headers=headers)
    return (json.loads(response.text).get('result'))


def open_doors(id, direction, user_name):
    doors = [0,'вход', 'выход']
    print_log(f'В {datetime.now().strftime("%D:%H:%M:%S")}  {doors[direction]}  {user_name}! ')
    if not (id == previous_state['id'] and direction == previous_state['direction']):
        previous_state['id'] = id
        previous_state['direction'] = direction
        if (passing(int(id), direction, f'камера {direction}') == 'ok'):  # моя откравает проход
            pass
        else:
            exit()  # скорее всего просрочен токен авторизации выходим и заходим снова

    


def read_db1():
    db = {}
    url = f'http://{connection_config["host"]}/api/users/staff/list'
    querystring = {"withPhone": "true"}
    for id, name in pd.read_json(requests.request("get", url, headers=headers, params=querystring).text)[
        ['id', 'name']].values:
        db[id] = name
    print(db)
    return db


def read_db():
    db = {}
    url = f'http://{connection_config["host"]}/api/users/staff/list'
    querystring = {"withPhone": "true"}
    for user in json.loads(requests.request("get", url, headers=headers, params=querystring).text):
        db[user['id']] = user['name']
    return db


def read_excel(file):
    df = pd.read_excel(file)
    user_dict = {ind: name for ind, name in df.values}
    return user_dict


def get_encodings(dict_users, encoded_path):
    id_to_encoding = {}
    for id in dict_users.keys():
        if id == 0:
            continue
        path = f'{encoded_path}/{id}'
        with open(path, 'rb') as file:
            encoding = pickle.load(file)
        id_to_encoding[id] = encoding
    return id_to_encoding


def face_in_area(face_location, area):
    return area[0] < (face_location[3] + face_location[1]) // 2 < area[2] and area[1] < (
            face_location[0] + face_location[2]) // 2 < area[3]

def print_log(prn_text):
    o = open(log_file,'a')
    if config['test_mode']:
        print(prn_text)
    print(prn_text,file=o)
    o.close()
 
