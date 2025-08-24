from src.utils import read_excel, get_connection
import pickle

class FaceEngine:
    def __init__(self, config, users, camera):
        self.config = config
        self.embeddings = self.load_embeddings(users)


    def encode_image(self, image):
        pass

    def encode_folder(self):
        pass

    def detect_faces(self, frame):
        pass

    def add_person(self, name, image):
        pass

    def load_embeddings(self, users):    # get user faces' embeddings
        embeddings = {}

        for id in users.keys():
            if id == 0:
                continue
            path = f'{self.config["embedding_folder"]}/{id}'

            with open(path, 'rb') as file:
                embedding = pickle.load(file)

            embeddings[id] = embedding
        return embeddings
