from src.engines import FaceEngine, FacenetEngine

def get_engine(config, users, camera) -> FaceEngine:
    mode = config["mode"]

    if mode == "facenet":
        return FacenetEngine(config, users, camera)
    elif mode == "face_recognition":
        raise NotImplementedError()
    else:
        raise ValueError(f"Unknown mode: {mode}")
