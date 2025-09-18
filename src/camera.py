import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tenacity import retry, stop_after_attempt, wait_fixed

from src.utils import DoorState

STOP_AFTER_ATTEMPT = 200
WAIT = 0.1


class Camera:
    def __init__(self, config, stop_event):
        self.config = config
        self.camera_config = config["camera"]
        self.camera_id = self.camera_config["id"]
        self.reduce_frame = self.camera_config["reduce_frame"]
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.stop_event = stop_event

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) / self.reduce_frame)
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / self.reduce_frame)

        self.frame = None
        self.show_frame = None
        self.video_capture()
        self.exit_area, self.entrance_area = self.get_frame_areas()

    @retry(wait=wait_fixed(WAIT), stop=stop_after_attempt(STOP_AFTER_ATTEMPT))  # retry if attempt failed
    def video_capture(self):
        ret, new_frame = self.cap.read()
        if ret:
            self.frame = cv2.resize(new_frame, (self.frame_width, self.frame_height))
        else:
            raise IOError("Camera is not available")
        return ret, self.frame

    def generate(self):
        try:
            while not self.stop_event.is_set():
                if self.show_frame is not None:
                    _, jpeg = cv2.imencode('.jpg', self.show_frame)
                else:
                    _, jpeg = cv2.imencode('.jpg', np.ndarray((3, 3, 3), dtype=np.uint8))
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                time.sleep(0.05)
        except GeneratorExit:
            pass

    def get_frame_areas(self):  # get entrance and exit areas coordinates
        area_1 = self.config['turnstiles']['area_1']
        area_2 = self.config['turnstiles']['area_2']

        exit_area = [int(area_1[0] * self.frame_width), int(area_1[1] * self.frame_height),
                     int((area_1[0] + area_1[2]) * self.frame_width),
                     int((area_1[1] + area_1[3]) * self.frame_height)]
        entrance_area = [int(area_2[0] * self.frame_width), int(area_2[1] * self.frame_height),
                         int((area_2[0] + area_2[2]) * self.frame_width),
                         int((area_2[1] + area_2[3]) * self.frame_height)]

        return exit_area, entrance_area

    def face_in_area(self, face_location, area):
        if self.camera_config['frame_mode'] == 'center':
            return area[0] < (face_location[0] + face_location[2]) // 2 < area[2] and area[1] < (
                    face_location[1] + face_location[3]) // 2 < area[3]
        else:
            return area[0] < face_location[0] and face_location[3] < area[3] and area[1] < face_location[1] and \
                face_location[2] < area[2]

    def check_areas(self, face_locations, user_ids):
        for i in range(len(face_locations)):
            if user_ids[i] == 0:
                continue
            if self.face_in_area(face_locations[i], self.exit_area):
                return i, DoorState.EXIT
            elif self.face_in_area(face_locations[i], self.entrance_area):
                return i, DoorState.ENTRANCE
        return None, DoorState.CLOSED

    def show(self, face_locations, recognized_ids, users):
        # convert BGR -> RGB PIL
        img = Image.fromarray(self.frame[:, :, ::-1].copy()).convert("RGB")

        img = self.draw_box(img, self.exit_area, label='Exit', color='red')
        img = self.draw_box(img, self.entrance_area, label='Entrance', color='green')

        for user_id, box in zip(recognized_ids, face_locations):
            if user_id not in users:
                user_id = 0
            user_name = users[user_id]
            img = self.draw_box(img, box, label=user_name)

        self.show_frame = np.asarray(img)[:, :, ::-1]

    def draw_box(
            self,
            image: np.ndarray,
            box,
            label=None,
            color=(66, 133, 244),  # nice blue
            outline=3,  # outline width
            radius=12,  # corner radius
            label_text_color=(255, 255, 255),
            font_path=None,  # e.g. "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            font_size=16
    ):
        """
        Draws pretty rounded boxes + label pills on top of image (OpenCV-style).
        """

        # overlay for semi-transparency
        draw = ImageDraw.Draw(image, "RGB")

        # optional font
        try:
            font = ImageFont.truetype(font_path, font_size) if font_path else ImageFont.load_default()
        except Exception:
            font = ImageFont.load_default()

        x1, y1, x2, y2 = box
        x1 = np.clip(x1, 0, self.frame_width)
        x2 = np.clip(x2, 0, self.frame_width)
        y1 = np.clip(y1, 0, self.frame_height)
        y2 = np.clip(y2, 0, self.frame_height)

        # --- filled rounded rect ---
        draw.rounded_rectangle(
            [x1, y1, x2, y2],
            radius=radius,
            fill=None,
            outline=color,
            width=outline
        )

        # --- label pill (optional) ---
        if label is not None:
            # text size
            tw, th = draw.textbbox((0, 0), label, font=font)[2:]
            pad_x, pad_y = 8, 4
            pill_h = th + 2 * pad_y
            pill_w = tw + 2 * pad_x
            pill_x1, pill_y1 = x1, max(y1 - pill_h - 6, 2)
            pill_x2, pill_y2 = pill_x1 + pill_w, pill_y1 + pill_h

            # pill background
            draw.rounded_rectangle(
                [pill_x1, pill_y1, pill_x2, pill_y2],
                radius=int(pill_h / 2),
                fill=None
            )
            # text
            draw.text(
                (pill_x1 + pad_x, pill_y1 + pad_y),
                label, font=font, fill=label_text_color
            )

        return image

    def release(self):
        if self.cap:
            self.cap.release()
        self.cap = None
