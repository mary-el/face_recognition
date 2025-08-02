import glob
import os
import pickle

import numpy as np
import tqdm

import face_recognition as fr
from src.utils import face_in_area


def encode_image(image, known_location=None):
    if type(image) != np.array:
        image = np.array(image)
    return fr.face_encodings(image, known_location, num_jitters=config['face-recognition']['num_jitters'])


def encode_folder():
    save_to = config['face-recognition']['embedding_folder']
    print(save_to)
    os.makedirs(save_to, exist_ok=True)
    for file in tqdm.tqdm(glob.glob(config['images_folder'] + '/*')):
        enc_file = f'{save_to}/{os.path.basename(file).split(".")[0]}'
        if not os.path.exists(enc_file):
            image = fr.load_image_file(file)
            encoding = encode_image(image)[0]
            with open(enc_file, 'wb') as f:
                pickle.dump(encoding, f)


def get_best_match_idx(encodings: list, face_encoding: np.array):
    matches = fr.compare_faces(encodings, face_encoding, tolerance=config['face-recognition']['tolerance'])
    face_distances = fr.face_distance(encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if not matches[best_match_index]:
        return None
    return best_match_index


def search_for_faces(id_to_encoding, small_frame, left_area, right_area):
    face_locations = fr.face_locations(small_frame)
    if len(face_locations) == 0:
        return [], [], -1, None
    # face_locations
    face_encodings = encode_image(small_frame, face_locations)
    face_locations = np.array(face_locations)
    face_locations[:, [1, 3]] = face_locations[:, [3, 1]]
    recognized_names = []
    open_door = -1
    open_name = None
    names = list(id_to_encoding.keys())
    encodings = list(id_to_encoding.values())
    for i, face_encoding in enumerate(face_encodings):
        name = 0
        best_match_index = get_best_match_idx(encodings, face_encoding)
        if best_match_index is not None:
            name = names[best_match_index]
            if face_in_area(face_locations[i], left_area):
                open_name = name
                open_door = 2
            elif face_in_area(face_locations[i], right_area):
                open_name = name
                open_door = 1
        recognized_names.append(name)
    return recognized_names, face_locations, open_door, open_name
