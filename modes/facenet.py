import glob
import os
import pickle

import numpy as np
import torch
import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from config import config
from utils import face_in_area

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

mtcnn_scene = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True,
    device=device)
mtcnn_encode = MTCNN(
        image_size=160, margin=0, min_face_size=20,
        thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval()


def encode_image(image):
    img_cropped = mtcnn_encode(image)
    encoding = resnet(img_cropped.unsqueeze(0).to(device)).detach().cpu()
    return [encoding]


def encode_folder():
    save_to = config['facenet']['embedding_folder']
    os.makedirs(save_to, exist_ok=True)
    for file in tqdm.tqdm(glob.glob(config['images_folder'] + '/*')):
        enc_file = f'{save_to}/{os.path.basename(file).split(".")[0]}'
        img = Image.open(file)
        if not os.path.exists(enc_file):
            encoding = encode_image(img)[0]
            with open(enc_file, 'wb') as f:
                pickle.dump(encoding, f)


def get_best_match_idx(encodings, embedding):
    distances = [(embedding - e).norm().item() for e in encodings]
    best_match_index = np.argmin(distances)
    if distances[best_match_index] >= config['facenet']['threshold']:
        return None
    return best_match_index


def search_for_faces(id_to_encoding, frame, left_area, right_area):
    face_locations, _ = mtcnn_scene.detect(frame, landmarks=False)
    recognized_names = []
    open_door = -1
    open_name = None
    if face_locations is None or len(face_locations) == 0:
        return [], [], open_door, open_name
    try:
        faces = mtcnn_scene.extract(frame, face_locations, save_path='faces/temp.jpg')
    except:
        return [], [], open_door, open_name

    face_locations = np.array(face_locations)
    face_locations[:, [0, 1, 2, 3]] = face_locations[:, [1, 0, 3, 2]]
    faces = faces.to(device)
    img_embeddings = resnet(faces).detach().cpu()
    names = list(id_to_encoding.keys())
    encodings = list(id_to_encoding.values())
    for i, embedding in enumerate(img_embeddings):
        name = 0
        best_match_index = get_best_match_idx(encodings, embedding)
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
