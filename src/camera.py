import argparse
from datetime import datetime

import cv2

from config import config

area_1 = config['turnstiles']['area_1']
area_2 = config['turnstiles']['area_2']

from utils import set_headers, open_doors, read_db, read_excel, get_encodings, print_log

date_time_format = config['date_time_format']

if config['mode'] == 'face-recognition':
    from engines.face_recognition import encode_folder, search_for_faces

    mode_paranoid = config[config['mode']]['tolerance']
elif config['mode'] == 'facenet':
    from engines.facenet import encode_folder, search_for_faces

    mode_paranoid = config[config['mode']]['threshold']
else:
    print_log(f'Unsupported mode: {config["mode"]}')
    exit()


def show_recognized_faces(frame, face_locations, recognized_ids, reduce_frame, left_area, right_area):
    cv2.rectangle(frame, (left_area[0] * reduce_frame, left_area[1] * reduce_frame),
                  (left_area[2] * reduce_frame, left_area[3] * reduce_frame), (0, 255, 255), 2)
    cv2.rectangle(frame, (right_area[0] * reduce_frame, right_area[1] * reduce_frame),
                  (right_area[2] * reduce_frame, right_area[3] * reduce_frame), (255, 0, 0), 2)
    for (top, left, bottom, right), id in zip(face_locations, recognized_ids):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top, right, bottom, left = [int(x * reduce_frame) for x in (top, right, bottom, left)]
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(frame,
                    dict_users[id],
                    (left + 6, bottom - 6), font, 0.5,
                    (255, 255, 255), 1)
    return frame


def video_capture(id_to_encoding, camera, dict_users, reduce_frame=2, show=True, test=False):
    cap = cv2.VideoCapture(camera)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    _, frame = cap.read()
    width, height = frame.shape[1] / reduce_frame, frame.shape[0] / reduce_frame
    left_area = [int(area_1[0] * width), int(area_1[1] * height), int((area_1[0] + area_1[2]) * width),
                 int((area_1[1] + area_1[3]) * height)]
    right_area = [int(area_2[0] * width), int(area_2[1] * height), int((area_2[0] + area_2[2]) * width),
                  int((area_2[1] + area_2[3]) * height)]
    stop = False
    while not stop:
        ret, frame = cap.read()
        small_frame = cv2.resize(frame, (0, 0), fx=1 / reduce_frame, fy=1 / reduce_frame)

        recognized_ids, face_locations, open_door, user_ind = search_for_faces(id_to_encoding, small_frame, left_area,
                                                                               right_area)
        if open_door != -1:
            if test:
                print(f'Door #{open_door} opened for {", ".join(dict_users[i] for i in recognized_ids if i != 0)}')
            else:
                open_doors(user_ind, open_door, dict_users[user_ind])
            file_name = f'./{config["frame_folder"]}/{datetime.now().strftime(date_time_format)}_Id{user_ind}_direct{open_door}.jpg'
            cv2.imwrite(file_name, show_recognized_faces(frame, face_locations, recognized_ids, reduce_frame, left_area,
                                                         right_area))

        if show:
            cv2.imshow('Face recognition',
                       show_recognized_faces(frame, face_locations, recognized_ids, reduce_frame, left_area,
                                             right_area))
        if cv2.waitKey(1) == 13:  # 13 is the Enter Key
            stop = True
    # Release camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Face recognition')
    parser.add_argument('-e', '--encode', action="store_true")
    args = parser.parse_args()

    if args.encode:
        encode_folder()
        exit()

    if config['source'] == 'excel':
        dict_users = read_excel(config['excel_file'])
    else:
        set_headers()
        dict_users = read_db()

    dict_users[0] = config['no_name_user']
    id_to_encoding = get_encodings(dict_users, config[config['mode']]['embedding_folder'])

    print_log(
        f'started at {datetime.now().strftime("%D:%H:%M:%S")} mode={config["mode"]}, mode_paranoid={mode_paranoid}')
    video_capture(id_to_encoding, camera=config['camera'], show=config['show'], dict_users=dict_users,
                  test=config['test_mode'])
