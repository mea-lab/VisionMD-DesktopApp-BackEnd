import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base_detector import BaseDetector

class HandDetector(BaseDetector):
    """
    Detector for hand landmarks using MediaPipe's HandLandmarker.
    """

    def get_detector(self) -> vision.HandLandmarker:
        running_mode = vision.RunningMode
        current_dir = os.path.dirname(__file__)
        base_options = python.BaseOptions(
            model_asset_path=os.path.join(current_dir, '../models/hand_landmarker.task')
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            running_mode=running_mode.VIDEO
        )
        return vision.HandLandmarker.create_from_options(options=options)
