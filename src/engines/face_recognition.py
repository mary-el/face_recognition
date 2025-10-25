import numpy as np
from src.engines.base import FaceEngine


class FaceRecognitionEngine(FaceEngine):
    """Placeholder for face_recognition-based engine - NOT IMPLEMENTED"""
    
    def __init__(self, config, users, camera):
        """Initialize placeholder engine (always raises NotImplementedError)"""
        super().__init__(config, users, camera)
        raise NotImplementedError(
            "face_recognition engine is not currently working. "
            "Please use 'facenet' mode instead."
        )
    
    def detect_faces(self, frame: np.ndarray):
        """Placeholder method - not implemented"""
        raise NotImplementedError()
