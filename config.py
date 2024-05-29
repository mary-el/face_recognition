tolerance = 0.55  # в этой строке  tolerance, чем ниже, тем строже
num_jitters = 5  # Сколько раз нужно пересчитывать лицо при вычислении кодировки. Чем больше, тем точнее, но медленнее
host = "192.168.0.19"  # IP сервера с Perco
id_tur = 3  # id турникета
headers = {
    "Content-type": "application/json; charset=UTF-8",
    "Authorization": "Bearer null"
}
previous_state = {
    'id': -1,
    'direction': 0
}
NoName = '?неопределенный'


X0, Y0, W0, H0 = 0.1, 0.027, 0.2, 0.87
X1, Y1, W1, H1 = 0.35, 0.02, 0.2, 0.87