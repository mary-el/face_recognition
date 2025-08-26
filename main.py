import threading
import time
from contextlib import asynccontextmanager
from typing import Dict

import yaml
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from src.camera import Camera
from src.engines import get_engine, FaceEngine
from src.utils import Connection, DoorState, connection, load_users, setup_logger

templates = Jinja2Templates(directory="templates")

CONFIG_FILE = "configs/config_current.yaml"
config = {}
face_engine: FaceEngine
connection: Connection
camera: Camera
users: Dict[str, str]  # user ID -> name
logger = None


def init(config):
    global face_engine, camera, connection, users, logger

    logger = setup_logger(config)
    logger.info(f"Updating config")
    connection = Connection(config)
    users = load_users(config)
    camera = Camera(config)
    face_engine = get_engine(config, users, camera)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config

    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    init(config)
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
            open_n, door_state = camera.check_areas(face_locations, user_ids)
            if door_state != DoorState.CLOSED:
                logger.info(f'Door {door_state} opened for {users[user_ids[open_n]]}')
                if config["test_mode"]:
                    print(f'Door {door_state} opened for {users[user_ids[open_n]]}')
                else:
                    connection.open_doors(user_ids[open_n], door_state, users[user_ids[open_n]])
            camera.show(face_locations, user_ids, users)
        time.sleep(0.01)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("camera.html", {"request": request})


@app.post("/config")
async def set_config(config_file: UploadFile = File(...)):
    global config

    try:
        raw_data = await config_file.read()
        config = yaml.safe_load(raw_data)
        init(config)

        with open(CONFIG_FILE, "wb") as f:
            f.write(raw_data)

    except yaml.YAMLError as e:
        raise HTTPException(status_code=422, detail=f"Invalid YAML content: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

    return {"filename": config_file.filename, "message": "Config updated"}


@app.post("/sync")
async def sync():
    try:
        init(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    return {"message": "Data synchronized"}


@app.get("/video_feed")
def video_feed():
    try:
        return StreamingResponse(camera.generate(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        return {"error": str(e)}
