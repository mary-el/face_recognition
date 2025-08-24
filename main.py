import argparse
import threading
import time
from typing import Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.camera import Camera
from src.config import load_config
from src.engines import get_engine, FaceEngine
from src.utils import Connection, DoorState, connection, load_users

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

config = {}
face_engine: FaceEngine
connection: Connection
camera: Camera
users: Dict[str, str]   # user ID -> name



@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, face_engine, camera, connection, users
    config_file="configs/config.yaml"
    config = load_config(config_file)  # get config from the file
    connection = Connection(config)
    users = load_users(config)
    camera = Camera(config)
    face_engine = get_engine(config, users, camera)
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        camera.release()

app = FastAPI(lifespan=lifespan)


def capture_loop():
    while True:
        ret, new_frame = camera.video_capture()
        if ret:
            frame = new_frame
            user_ids, face_locations = face_engine.detect_faces(frame)
            open_n, door_state = camera.check_areas(face_locations)
            if door_state != DoorState.CLOSED:
                if config["test"]:
                    print(f'Door #{door_state} opened for {users[user_ids[open_n]]}')
                else:
                    connection.open_doors(user_ids[open_n], door_state, users[user_ids[open_n]])
            camera.show(face_locations, user_ids, users)
        time.sleep(0.1)


@app.get("/")
def home():
    return {"Hello": "World"}


# @app.post("/config")
# def set_config(config_file: str = Form(...)):
#     load_config(config_file)
#     return {"status": "updated"}
#
#
# @app.post("/add_person")
# def add_person(image: UploadFile, name: str = Form(...)):
#     return face_engine.add_person(image.file, name)


# @app.post("/sync_db")
# def sync_db():
#     face_engine.sync_database()
#     return {"status": "synced"}

@app.get("/video_feed")
def video_feed():
    try:
        return StreamingResponse(camera.generate(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
            return {"error": str(e)}
