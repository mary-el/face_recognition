from src.engines import FaceEngine, FacenetEngine, FaceRecognitionEngine

def get_engine(config, users, camera) -> FaceEngine:
    """Factory function to create appropriate face recognition engine"""
    mode = config["mode"]

    if mode == "facenet":
        return FacenetEngine(config, users, camera)
    elif mode == "face_recognition":
        return FaceRecognitionEngine(config, users, camera)
    else:
        raise ValueError(f"Unknown mode: {mode}")
