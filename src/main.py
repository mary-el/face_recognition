import argparse
import threading
import time

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.camera import Camera
from src.config import load_config
from src.engines import get_engine, FaceEngine
from src.utils import Connection

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

config = {}
face_engine: FaceEngine
camera: Camera


def capture_loop():
    while True:
        ret, new_frame = camera.video_capture()
        if ret:
            frame = new_frame
            recognized_names, face_locations, door_state, open_name = face_engine.detect_faces(frame)
            ### TODO
        time.sleep(0.01)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("config_ui.html", {"request": request})


@app.post("/config")
def set_config(config_file: str = Form(...)):
    load_config(config_file)
    return {"status": "updated"}


@app.post("/add_person")
def add_person(image: UploadFile, name: str = Form(...)):
    return face_engine.add_person(image.file, name)


# @app.post("/sync_db")
# def sync_db():
#     face_engine.sync_database()
#     return {"status": "synced"}

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(camera.generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.on_event("startup")
def start():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="set config file",
        type=str,
        default="configs/config.yaml",
    )
    args = parser.parse_args()
    global config, face_engine, camera

    config = load_config(args.config)  # get config from the file
    connection = Connection(config)
    camera = Camera(config)
    face_engine = get_engine(config)
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
