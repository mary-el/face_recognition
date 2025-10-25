import pickle


class FaceEngine:
    """Base class for face recognition engines"""
    
    def __init__(self, config, users, camera):
        self.config = config
        self.embeddings = self.load_embeddings(users)

    def encode_image(self, image):
        """Encode image to face embedding"""
        pass

    def encode_folder(self):
        """Generate embeddings for all images in folder"""
        pass

    def detect_faces(self, frame):
        """Detect faces in frame and return recognized IDs and locations"""
        pass

    def load_embeddings(self, users):
        """Load face embeddings for all users"""
        embeddings = {}

        for id in users.keys():
            if id == 0:  # Skip unknown user placeholder
                continue
            path = f'{self.config["embedding_folder"]}/{id}'

            with open(path, 'rb') as file:
                embedding = pickle.load(file)

            embeddings[id] = embedding
        return embeddings
