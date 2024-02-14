import cv2
import face_recognition as fr
import numpy as np
import pandas as pd


def show_recognized_faces(frame, face_locations, recognized_names, reduce_frame):
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


def open_doors():
    print('Open the door!')


def video_capture(encodings, names, reduce_frame=2, show=True, link=0):
    cap = cv2.VideoCapture(link)
    frame_n = 0
    while True:
        ret, frame = cap.read()
        if frame is None:
            continue
        frame_n += 1
        if frame_n != 3:
            continue
        frame_n = 0
        small_frame = cv2.resize(frame, (0, 0), fx=1 / reduce_frame, fy=1 / reduce_frame)
        face_locations = fr.face_locations(small_frame)
        face_encodings = fr.face_encodings(small_frame, face_locations)

        recognized_names = []
        found_match = False
        for face_encoding in face_encodings:
            matches = fr.compare_faces(encodings, face_encoding)
            name = '?'
            face_distances = fr.face_distance(encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = names[best_match_index]
                found_match = True
            recognized_names.append(name)
        if found_match:
            open_doors()
        if show:
            cv2.imshow('Face recognition', show_recognized_faces(frame, face_locations, recognized_names, reduce_frame))
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            break

    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()


def read_db(file):
    df = pd.read_excel(file)
    return df


def encode_faces(file):
    df = read_db(file)
    encodings = []
    names = []
    for file, name in df.values:
        path = 'images/' + file
        image = fr.load_image_file(path)
        encodings.append(fr.face_encodings(image)[0])
        names.append(name)
    return encodings, names


if __name__ == '__main__':
    encodings, names = encode_faces('db.xlsx')
    video_capture(encodings, names, link=0)
