from src.utils import read_excel, get_connection
import pickle

class FaceEngine:
    def __init__(self, config):
        self.config = config
        self.users = self.load_users()
        self.embeddings = self.load_embeddings()


    def encode_image(self, image):
        pass

    def encode_folder(self):
        pass

    def detect_faces(self, frame):
        pass

    def add_person(self, name, image):
        pass

    def load_users(self):   # loading users' names and ids
        if self.config['source'] == 'excel':
            users = read_excel(self.config['excel_file'])
        else:
            connection = get_connection(self.config)
            users = connection.read_users()
        users[0] = self.config['no_name_user']
        return users

    def load_embeddings(self):    # get user faces' embeddings
        embeddings = {}
        for id in self.users.keys():
            if id == 0:
                continue
            path = f'{self.config["embedding_folder"]}/{id}'

            with open(path, 'rb') as file:
                embedding = pickle.load(file)

            embeddings[id] = embedding
        return embeddings

    def get_frame_areas(self, frame):   # get entrance and exit areas coordinates
        area_1 = self.config['turnstiles']['area_1']
        area_2 = self.config['turnstiles']['area_2']

        width, height = frame.shape[1], frame.shape[0]

        exit_area = [int(area_1[0] * width), int(area_1[1] * height), int((area_1[0] + area_1[2]) * width),
                     int((area_1[1] + area_1[3]) * height)]
        entrance_area = [int(area_2[0] * width), int(area_2[1] * height), int((area_2[0] + area_2[2]) * width),
                      int((area_2[1] + area_2[3]) * height)]
        return exit_area, entrance_area

    def face_in_area(self, face_location, area):
        if self.config['frame_mode'] == 'center':
            return area[0] < (face_location[3] + face_location[1]) // 2 < area[2] and area[1] < (
                    face_location[0] + face_location[2]) // 2 < area[3]
        else:
            return area[0] < face_location[1] and face_location[3] < area[2] and area[1] < face_location[0] and \
                face_location[2] < area[3]
