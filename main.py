import signal
import threading
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
stop_event = threading.Event()  # graceful shutdown event
thread = None


def init(config):
    """Initialize all components with given config"""
    global face_engine, camera, connection, users, logger, stop_event

    logger = setup_logger(config)
    logger.info("Updating config")
    connection = Connection(config)
    users = load_users(config)
    camera = Camera(config, stop_event)
    face_engine = get_engine(config, users, camera)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global config, stop_event, thread

    with open(CONFIG_FILE, "r") as file:
        config = yaml.safe_load(file)

    init(config)
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    try:
        yield
    finally:
        logger.info("App stopped")


app = FastAPI(lifespan=lifespan)


def handle_exit(*args):
    global stop_event, thread, camera

    logger.info("Shutting down...")
    stop_event.set()
    if thread and thread.is_alive():
        thread.join(timeout=2)
    if camera:
        camera.release()

    orig_handler(*args)


orig_handler = signal.signal(signal.SIGINT, handle_exit)


def capture_loop():
    """Main capture loop - processes frames and controls doors"""
    while not stop_event.is_set():
        try:
            ret, new_frame = camera.video_capture()
        except Exception as exc:
            logger.exception("Error capturing frame: %s", exc)
            continue

        if not ret:
            continue

        try:
            frame = new_frame
            user_ids, face_locations = face_engine.detect_faces(frame)
        except Exception as exc:
            logger.exception("Error running face detection: %s", exc)
            continue

        try:
            open_n, door_state = camera.check_areas(face_locations, user_ids)

            if door_state != DoorState.CLOSED and open_n is not None:
                user_id = user_ids[open_n]
                user_name = users.get(user_id, users.get(0, "Unknown"))
                logger.info(f'Door {door_state} opened for {user_name}')
                if config.get("test_mode"):
                    print(f'Door {door_state} opened for {user_name}')
                else:
                    connection.open_doors(user_id, door_state, user_name)
            camera.show(face_locations, user_ids, users)
        except Exception as exc:
            logger.exception("Error processing frame overlay or door logic: %s", exc)
    logger.info("Loop exited")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("camera.html", {"request": request})


@app.post("/config")
async def set_config(config_file: UploadFile = File(...)):
    """Upload and apply new configuration"""
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
    """Synchronize users and embeddings from database"""
    try:
        init(config)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")
    return {"message": "Data synchronized"}


@app.get("/video_feed")
def video_feed():
    """Stream video feed to web interface"""
    try:
        return StreamingResponse(camera.generate(), media_type="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        return {"error": str(e)}
