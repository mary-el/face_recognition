from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request
from src.config import load_config
from src.engines import get_engine, FaceEngine
import argparse

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

config = {}
face_engine: FaceEngine

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

@app.post("/detect")
def detect(image: UploadFile):
    return face_engine.detect(image.file)

@app.post("/sync_db")
def sync_db():
    face_engine.sync_database()
    return {"status": "synced"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        help="set config file",
        type=str,
        default="configs/config.yaml",
    )
    args = parser.parse_args()
    global config, face_engine

    config = load_config(args.config)  # get config from the file
    face_engine = get_engine(config)
