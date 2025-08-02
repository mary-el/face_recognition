import glob
import os
import pickle

import numpy as np
import torch
import tqdm
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

from src.engines.base import FaceEngine
from src.utils import DoorState

class FacenetEngine(FaceEngine):
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.resnet = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()   # face embedding model

        self.mtcnn = MTCNN(   # model for scene face detection
            image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True, keep_all=True,
            device=self.device)

        # self.mtcnn_encode = MTCNN(  # model for
        #     image_size=160, margin=0, min_face_size=20,
        #     thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
        #     device=self.device)

    def encode_image(self, image):
        img_cropped = self.mtcnn(image)
        encoding = self.resnet(img_cropped.to(self.device)).detach().cpu()
        return [encoding]


    def encode_folder(self):
        save_to = self.config['facenet']['embedding_folder']
        os.makedirs(save_to, exist_ok=True)
        for file in tqdm.tqdm(glob.glob(self.config['images_folder'] + '/*')):
            enc_file = f'{save_to}/{os.path.basename(file).split(".")[0]}'
            if not os.path.exists(enc_file):
                img = Image.open(file)
                encoding = self.encode_image(img)[0]
                with open(enc_file, 'wb') as f:
                    pickle.dump(encoding, f)


    def get_best_match_idx(self, embeddings, face_embedding):
        distances = [(face_embedding - e).norm().item() for e in embeddings]
        best_match_index = np.argmin(distances)
        if distances[best_match_index] >= self.config['facenet']['threshold']:
            return None
        return best_match_index


    def detect_faces(self, frame: np.ndarray):
        exit_area, entrance_area = self.get_frame_areas(frame)
        recognized_names = []
        door_state = DoorState.CLOSED
        open_name = None

        face_locations, _ = self.mtcnn.detect(frame, landmarks=False)
        if face_locations is None or len(face_locations) == 0:  # no faces found
            return [], [], door_state, open_name
        try:
            faces = self.mtcnn.extract(frame, face_locations, save_path='faces/temp.jpg')
        except:
            return [], [], door_state, open_name

        face_locations = np.array(face_locations)
        face_locations[:, [0, 1, 2, 3]] = face_locations[:, [1, 0, 3, 2]]

        faces = faces.to(self.device)
        img_embeddings = self.resnet(faces).detach().cpu()

        user_ids = list(self.embeddings.keys())
        embeddings = list(self.embeddings.values())
        open_id = 0

        for i, face_embedding in enumerate(img_embeddings):
            idx = 0
            best_match_index = self.get_best_match_idx(embeddings, face_embedding)    # getting matches for each found face

            if best_match_index is not None:
                idx = user_ids[best_match_index]
                if self.face_in_area(face_locations[i], exit_area):
                    open_id = idx
                    door_state = DoorState.EXIT
                elif self.face_in_area(face_locations[i], entrance_area):
                    open_id = idx
                    door_state = DoorState.ENTRANCE
            recognized_names.append(self.users[idx])

        return recognized_names, face_locations, door_state, self.users[open_id]
