import math
import cv2
import mediapipe as mp
import numpy as np
import os, uuid, time, json, traceback
from django.core.files.storage import FileSystemStorage
from django.conf import settings

from .base_task import BaseTask
from app.analysis.detectors.mp_poseheavy_detector import PoseHeavyDetector
from app.analysis.signal_analyzers.peakfinder_signal_analyzer import PeakfinderSignalAnalyzer

class LegAgilityLeftTask(BaseTask):
    """
    Leg Agility task for the left leg.
    Tracks shoulder midpoint, left knee, and hip midpoint.
    Signal is the vertical difference (shoulder y-coordinate minus knee y-coordinate).
    Normalization factor is the average distance between shoulder midpoint and hip midpoint.
    """
    LANDMARKS = {
        "NOSE": 0,
        "LEFT_EYE_INNER": 1,
        "LEFT_EYE": 2,
        "LEFT_EYE_OUTER": 3,
        "RIGHT_EYE_INNER": 4,
        "RIGHT_EYE": 5,
        "RIGHT_EYE_OUTER": 6,
        "LEFT_EAR": 7,
        "RIGHT_EAR": 8,
        "MOUTH_LEFT": 9,
        "MOUTH_RIGHT": 10,
        "LEFT_SHOULDER": 11,
        "RIGHT_SHOULDER": 12,
        "LEFT_ELBOW": 13,
        "RIGHT_ELBOW": 14,
        "LEFT_WRIST": 15,
        "RIGHT_WRIST": 16,
        "LEFT_HIP": 23,
        "RIGHT_HIP": 24,
        "LEFT_KNEE": 25,
        "RIGHT_KNEE": 26,
        "LEFT_ANKLE": 27,
        "RIGHT_ANKLE": 28,
        "LEFT_HEEL": 29,
        "RIGHT_HEEL": 30,
        "LEFT_FOOT_INDEX": 31,
        "RIGHT_FOOT_INDEX": 32
    }

    def __init__(self):
        self.video_id = None
        self.video_fps = None
        self.video_rotation = None
        self.video_file_path = None
        self.video_file_name = None
        
        self.task_name = None
        self.task_norm_strategy = None
        self.task_start_time = None
        self.task_start_frame_idx = None
        self.task_end_time = None
        self.task_end_frame_idx = None

        self.original_bounding_box = None
        self.enlarged_bounding_box = None
        self.subject_bounding_boxes = None

    def api_response(self, request):
        try:
            # 1) Process video and define all abstract class parameters
            self.prepare_video_parameters(request)

            # 3) Get analyzer
            signal_analyzer = self.get_signal_analyzer()

            # 4) Extract landmarks using the defined detector
            result = self.extract_landmarks()
            essential_landmarks, all_landmarks = result

            # 5) Compute normalization factor.
            normalization_factor = self.calculate_normalization_factor(essential_landmarks)

            # 6) Compute the raw 1D signal (vertical difference between shoulder and knee).
            raw_signal = self.calculate_signal(essential_landmarks)

            # 7) Get output from the signal analyzer
            results = signal_analyzer.analyze(
                normalization_factor=normalization_factor,
                raw_signal=raw_signal,
                start_time=self.task_start_time,
                end_time=self.task_end_time
            )
            
            # 6) Structure output
            output = {}
            output['File name'] = self.video_file_name
            output['Task name'] = self.task_name
            output = output | results
            output["landMarks"] = essential_landmarks
            output["allLandMarks"] = all_landmarks
            output["normalization_factor"] = normalization_factor

        except Exception as e:
            raise Exception(str(e))

        return output

    def prepare_video_parameters(self, request):
        """
        Parses POST data, saves the video file, computes bounding boxes and frame indices.
        """
        video_id = request.GET.get('id', None)
        if not video_id:
            raise Exception("Video project id not provided.")
        
        json_raw = request.POST.get('json_data')
        if not json_raw:
            raise Exception("Missing 'json_data' in POST data")
        
        try:
            json_data = json.loads(json_raw)
        except json.JSONDecodeError:
            raise Exception("Invalid JSON in 'json_data'")
        
        folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads", video_id)
        if not os.path.isdir(folder_path):
            raise Exception("Video project folder does not exist.")

        metadata_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file for video does not exist.")
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
        except Exception as e:
            raise Exception(f"Metadata file '{metadata_path}' cannot be decoded {e}")              
    
        #  Prepare video data
        video_fps = metadata["metadata"]["fps"]
        video_rotation = metadata["metadata"]["rotation"]
        video_file_name = metadata["metadata"]["video_name"]
        video_file_path = os.path.join(settings.MEDIA_ROOT, "video_uploads", video_id, video_file_name)
        video_width, video_height = BaseTask.get_video_width_height(video_file_path, video_rotation)

        #  Prepare task data
        task_name = json_data["task_name"]
        task_start_time = json_data['start_time']
        task_end_time = json_data['end_time']
        task_start_frame_idx = round(video_fps * task_start_time)
        task_end_frame_idx = round(video_fps * task_end_time)

        # Prepare bounding box data
        original_bounding_box = json_data['boundingBox']
        enlarged_bounding_box = {
            'x': int(max(0, original_bounding_box['x'] - original_bounding_box['width'] * 0.125)),
            'y': int(max(0, original_bounding_box['y'] - original_bounding_box['height'] * 0.125)),
            'width': int(min(video_width - original_bounding_box['x'], original_bounding_box['width'] * 1.25)),
            'height': int(min(video_height - original_bounding_box['y'], original_bounding_box['height'] * 1.25)),
        }

        self.video_id = video_id
        self.video_fps = video_fps
        self.video_rotation = video_rotation
        self.video_file_name = video_file_name
        self.video_file_path = video_file_path

        self.task_name = task_name
        self.task_start_time = task_start_time
        self.task_end_time = task_end_time
        self.task_start_frame_idx = task_start_frame_idx
        self.task_end_frame_idx = task_end_frame_idx

        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box
    
    def get_detector(self):
        return PoseHeavyDetector().get_detector()

    def get_signal_analyzer(self):
        return PeakfinderSignalAnalyzer()
    
    def extract_landmarks(self) -> tuple:
        """
        Iterate through frames between start_frame_idx and end_frame_idx,
        crop each frame to the enlarged bounding box, run the pose detector,
        and extract the essential landmarks needed for the leg agility signal.
        """

        detector = PoseHeavyDetector().get_detector()

        # Calculate coordinates for cropping.
        x1 = self.enlarged_bounding_box['x']
        y1 = self.enlarged_bounding_box['y']
        x2 = x1 + self.enlarged_bounding_box['width']
        y2 = y1 + self.enlarged_bounding_box['height']
        enlarged_coords = (x1, y1, x2, y2)

        ox1 = self.original_bounding_box['x']
        oy1 = self.original_bounding_box['y']
        ox2 = self.original_bounding_box['x'] + self.original_bounding_box['width']
        oy2 = self.original_bounding_box['y'] + self.enlarged_bounding_box['height']
        original_coords = (ox1,oy1,ox2,oy2)

        essential_landmarks = []
        all_landmarks = []

        video = cv2.VideoCapture(self.video_file_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, self.task_start_frame_idx)
        current_frame_idx = self.task_start_frame_idx

        while current_frame_idx < self.task_end_frame_idx:
            success, frame = video.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = BaseTask.correct_frame_orientation(rgb_frame,self.video_rotation)
            cropped_frame = rgb_frame[y1:y2, x1:x2, :].astype(np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_frame)

            timestamp = int((current_frame_idx / self.video_fps) * 1000)
            detection_result = detector.detect_for_video(image, timestamp)

            if not detection_result.pose_landmarks:
                essential_landmarks.append([])
                all_landmarks.append([])
            else:
                landmarks = detection_result.pose_landmarks[0]
                # Use the left knee since this is the left task.
                knee_idx = LegAgilityLeftTask.LANDMARKS["LEFT_KNEE"]
                knee = [
                    landmarks[knee_idx].x * (x2 - x1) + x1,
                    landmarks[knee_idx].y * (y2 - y1) + y1
                ]
                # Average the left and right shoulder coordinates.
                left_shoulder = landmarks[LegAgilityLeftTask.LANDMARKS["LEFT_SHOULDER"]]
                right_shoulder = landmarks[LegAgilityLeftTask.LANDMARKS["RIGHT_SHOULDER"]]
                shoulder_mid = [
                    ((left_shoulder.x + right_shoulder.x) / 2) * (x2 - x1) + x1,
                    ((left_shoulder.y + right_shoulder.y) / 2) * (y2 - y1) + y1
                ]
                # Average the left and right hip coordinates.
                left_hip = landmarks[LegAgilityLeftTask.LANDMARKS["LEFT_HIP"]]
                right_hip = landmarks[LegAgilityLeftTask.LANDMARKS["RIGHT_HIP"]]
                hip_mid = [
                    ((left_hip.x + right_hip.x) / 2) * (x2 - x1) + x1,
                    ((left_hip.y + right_hip.y) / 2) * (y2 - y1) + y1
                ]
                essential = [shoulder_mid, knee, hip_mid]
                all_lms = BaseTask.get_all_landmarks_coord(landmarks, enlarged_coords, original_coords)
                essential_landmarks.append(essential)
                all_landmarks.append(all_lms)
            current_frame_idx += 1

        video.release()
        detector.close()

        missing_percent = sum(1 for x in essential_landmarks if not x) / len(essential_landmarks)
        if missing_percent > 0.1:
            raise Exception((f"Left leg could not be found in more than 10% of the frames. The video quality may be too low or the video may not be a leg agility task."))

        return essential_landmarks, all_landmarks

    def calculate_signal(self, essential_landmarks):
        """
        For each frame, compute the difference (shoulder y - knee y).
        If no landmarks were detected in a frame, reuse the previous value.
        """
        signal = []
        prev_signal = 0
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 3:
                signal.append(prev_signal)
                continue
            shoulder_mid, knee, _ = frame_lms
            diff = abs(shoulder_mid[1] - knee[1])
            prev_signal = diff
            signal.append(diff)
        return signal

    def calculate_normalization_factor(self, essential_landmarks):
        """
        Compute the average distance between the shoulder midpoint and hip midpoint.
        """
        distances = []
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 3:
                continue
            shoulder_mid, _, hip_mid = frame_lms
            d = math.dist(shoulder_mid, hip_mid)
            distances.append(d)
        return float(np.mean(distances)) if distances else 1.0
