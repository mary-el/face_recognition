import argparse
import glob
import os
import pickle

import cv2
import face_recognition as fr
import numpy as np
import pandas as pd

X0, Y0, W0, H0 = 0.1, 0.1, 0.35, 0.8
X1, Y1, W1, H1 = 0.55, 0.1, 0.35, 0.8


def show_recognized_faces(frame, face_locations, recognized_names, reduce_frame, left_area, right_area):
    cv2.rectangle(frame, (left_area[0], left_area[1]), (left_area[2], left_area[3]), (0, 255, 255), 2)
    cv2.rectangle(frame, (right_area[0], right_area[1]), (right_area[2], right_area[3]), (255, 0, 0), 2)
    for (top, right, bottom, left), name in zip(face_locations, recognized_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= reduce_frame
        right *= reduce_frame
        bottom *= reduce_frame
        left *= reduce_frame

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
    return frame


def open_doors(name, door_number):
    doors = ['left', 'right']
    print(f'Open the {doors[door_number]} door for {name}!')


def video_capture(encodings, names, camera, reduce_frame=2, show=True):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _, frame = cap.read()
    width, height = frame.shape[1], frame.shape[0]
    left_area = [int(X0 * width), int(Y0 * height), int((X0 + W0) * width), int((Y0 + H0) * height)]
    right_area = [int(X1 * width), int(Y1 * height), int((X1 + W1) * width), int((Y1 + H1) * height)]
    stop = False
    while not stop:
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=1 / reduce_frame, fy=1 / reduce_frame)
        face_locations = fr.face_locations(small_frame)
        face_encodings = fr.face_encodings(small_frame, face_locations)

        recognized_names = []
        open_door = -1
        open_name = None
        for i, face_encoding in enumerate(face_encodings):
            matches = fr.compare_faces(encodings, face_encoding)
            name = '?'
            face_distances = fr.face_distance(encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = names[best_match_index]
                if (left_area[0] < (face_locations[i][3] + face_locations[i][1]) // 2 * reduce_frame < left_area[2] and
                        left_area[1] < (face_locations[i][0] + face_locations[i][2]) // 2 * reduce_frame < left_area[
                            3]):
                    open_name = name
                    open_door = 0
                elif (right_area[0] < (face_locations[i][3] + face_locations[i][1]) // 2 * reduce_frame < right_area[
                    2] and
                      right_area[1] < (face_locations[i][0] + face_locations[i][2]) // 2 * reduce_frame < right_area[
                          3]):
                    open_name = name
                    open_door = 1
            recognized_names.append(name)
        if open_door != -1:
            open_doors(open_name, open_door)
        if show:
            cv2.imshow('Face recognition',
                       show_recognized_faces(frame, face_locations, recognized_names, reduce_frame, left_area,
                                             right_area))
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            stop = True
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()


def read_db(file):
    df = pd.read_excel(file)
    return df


def encode_folder(folder, save_to):
    os.makedirs(save_to, exist_ok=True)
    for file in glob.glob(folder + '/*'):
        enc_file = f'{save_to}/{os.path.basename(file).split(".")[0]}'
        if not os.path.exists(enc_file):
            image = fr.load_image_file(file)
            encoding = fr.face_encodings(image)[0]
            with open(enc_file, 'wb') as f:
                pickle.dump(encoding, f)


def get_encodings(db_file, encoded_path):
    df = read_db(db_file)
    encodings = []
    names = []
    for file, name in df.values:
        path = f'{encoded_path}/{file}'
        with open(path, 'rb') as f:
            encoding = pickle.load(f)
        encodings.append(encoding)
        names.append(name)
    return encodings, names


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Face recognition')
    parser.add_argument('-e', '--encode', type=str)
    parser.add_argument('--db', type=str, default='db.xlsx')
    parser.add_argument('-p', '--path', type=str, default='encoded')
    parser.add_argument('-c', '--camera', type=str, default=0)
    parser.add_argument('-s', '--show', action='store_true')

    args = parser.parse_args()
    if args.encode:
        encode_folder(args.encode, args.path)
        exit()
    encodings, names = get_encodings(args.db, args.path)
    video_capture(encodings, names, camera=args.camera, show=args.show)
