import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .base_detector import BaseDetector

class PoseLandmarkerFullDetector(BaseDetector):
    """
    Detector for pose landmarks (full model) using MediaPipe's PoseLandmarker.
    """

    def get_detector(self) -> vision.PoseLandmarker:
        running_mode = vision.RunningMode
        current_dir = os.path.dirname(__file__)
        base_options = python.BaseOptions(
            model_asset_path=os.path.join(current_dir, '../models/pose_landmarker_full.task')
        )
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            output_segmentation_masks=False,
            running_mode=running_mode.VIDEO
        )
        return vision.PoseLandmarker.create_from_options(options=options)
