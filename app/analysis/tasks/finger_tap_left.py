import os
import cv2
import math
import json
import uuid
import mediapipe as mp
import numpy as np
import traceback
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from rest_framework.response import Response

from .base_task import BaseTask
from app.analysis.detectors.mp_hand_detector import HandDetector
from app.analysis.signal_analyzers.peakfinder_signal_analyzer import PeakfinderSignalAnalyzer

class FingerTapLeftTask(BaseTask):

# ------------------------------------------------------------------
# --- START: Abstract properties definitions
# ------------------------------------------------------------------
    LANDMARKS = {
        "WRIST": 0,
        "THUMB_CMC": 1,
        "THUMB_MCP": 2,
        "THUMB_IP": 3,
        "THUMB_TIP": 4,
        "INDEX_FINGER_MCP": 5,
        "INDEX_FINGER_PIP": 6,
        "INDEX_FINGER_DIP": 7,
        "INDEX_FINGER_TIP": 8,
        "MIDDLE_FINGER_MCP": 9,
        "MIDDLE_FINGER_PIP": 10,
        "MIDDLE_FINGER_DIP": 11,
        "MIDDLE_FINGER_TIP": 12,
        "RING_FINGER_MCP": 13,
        "RING_FINGER_PIP": 14,
        "RING_FINGER_DIP": 15,
        "RING_FINGER_TIP": 16,
        "PINKY_MCP": 17,
        "PINKY_PIP": 18,
        "PINKY_DIP": 19,
        "PINKY_TIP": 20
    }
# ------------------------------------------------------------------
# --- END: Abstract properties definitions
# ------------------------------------------------------------------





# -------------------------------------------------------------
# --- START: Abstract methods definitions
# -------------------------------------------------------------
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
            # 1. Define video parameters that were declared in init
            self.prepare_video_parameters(request)

            # 2. Extract landmarks from video
            essential_landmarks, all_landmarks = self.extract_landmarks()
            
            # 3. Calculate normalization factor from landmarks
            normalization_factor = self.calculate_normalization_factor(all_landmarks)

            # 4. Calculate signal from landmarks
            raw_signal = self.calculate_signal(essential_landmarks)
            
            # 5. Caclulate features from signal
            signal_analyzer = self.get_signal_analyzer()
            results = signal_analyzer.analyze(raw_signal, normalization_factor, self.task_start_time, self.task_end_time)
            
            # 6. Structure output
            output = {}
            output['File name'] = self.video_file_name
            output['Task name'] = self.task_name
            output = output | results
            output["landMarks"] = essential_landmarks
            output["allLandMarks"] = all_landmarks
            output["normalization_factor"] = normalization_factor

        except Exception as e:
            return Response(f"{e}", status=500)

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
        task_norm_strategy = json_data['norm_strategy']
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
        self.task_norm_strategy = task_norm_strategy

        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box

    def get_detector(self) -> object:
        return HandDetector().get_detector()

    def get_signal_analyzer(self) -> object:
        return PeakfinderSignalAnalyzer()

    def extract_landmarks(self) -> tuple:
        detector = HandDetector().get_detector()
        essential_landmarks = []
        all_landmarks = []
        enlarged_coords = (
            self.enlarged_bounding_box['x'],
            self.enlarged_bounding_box['y'],
            self.enlarged_bounding_box['x'] + self.enlarged_bounding_box['width'],
            self.enlarged_bounding_box['y'] + self.enlarged_bounding_box['height']
        )
        original_coords = (
            self.original_bounding_box['x'],
            self.original_bounding_box['y'],
            self.original_bounding_box['x'] + self.original_bounding_box['width'],
            self.original_bounding_box['y'] + self.original_bounding_box['height']
        )
        
        video = cv2.VideoCapture(self.video_file_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, self.task_start_frame_idx)
        current_frame_idx = self.task_start_frame_idx

        while current_frame_idx < self.task_end_frame_idx:
            success, frame = video.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = BaseTask.correct_frame_orientation(rgb_frame,self.video_rotation)
            x1, y1, x2, y2 = enlarged_coords
            image_data = rgb_frame[y1:y2, x1:x2, :].astype(np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
            timestamp = int(current_frame_idx / self.video_fps * 1000)
            detection_result = detector.detect_for_video(image, timestamp)
            
            # Look for the left hand
            hand_index = -1
            for idx, label in enumerate(detection_result.handedness):
                if label[0].category_name == "Left":
                    hand_index = idx
                    break

            if hand_index == -1 or not detection_result.hand_landmarks[hand_index]:
                video.release()
                detector.close()
                raise Exception(f"Left hand could not be found in frame {current_frame_idx}")
            else:
                hand_landmarks = detection_result.hand_landmarks[hand_index]
                thumb = BaseTask.get_landmark_coords(hand_landmarks[FingerTapLeftTask.LANDMARKS["THUMB_TIP"]], enlarged_coords, original_coords)
                index_finger = BaseTask.get_landmark_coords(hand_landmarks[FingerTapLeftTask.LANDMARKS["INDEX_FINGER_TIP"]], enlarged_coords, original_coords)
                middle_finger = BaseTask.get_landmark_coords(hand_landmarks[FingerTapLeftTask.LANDMARKS["MIDDLE_FINGER_TIP"]], enlarged_coords, original_coords)
                wrist = BaseTask.get_landmark_coords(hand_landmarks[FingerTapLeftTask.LANDMARKS["WRIST"]], enlarged_coords, original_coords)
                essential = [thumb, index_finger]
                all_lms = BaseTask.get_all_landmarks_coord(hand_landmarks, enlarged_coords, original_coords)
                essential_landmarks.append(essential)
                all_landmarks.append(all_lms)
            current_frame_idx += 1

        video.release()
        detector.close()
        return essential_landmarks, all_landmarks


    def calculate_signal(self, essential_landmarks) -> list:
        signal = []
        prev_dist = 0
        for frame_lms in essential_landmarks:
            if len(frame_lms) < 2:
                signal.append(prev_dist)
                continue
            thumb, index_finger = frame_lms[0], frame_lms[1]
            dist = math.dist(thumb, index_finger)
            prev_dist = dist
            signal.append(dist)
        return signal


    def calculate_normalization_factor(self, landmarks) -> float:
        LM = FingerTapLeftTask.LANDMARKS
        factors = []

        def has_idxs(frame, *idxs):
            return all(i < len(frame) for i in idxs)

        for frame in landmarks:
            # THUMB
            if self.task_norm_strategy == 'THUMBSIZE':
                if has_idxs(frame, 
                            LM['THUMB_CMC'], LM['THUMB_MCP'], 
                            LM['THUMB_IP'], LM['THUMB_TIP']):
                    d1 = math.dist(frame[LM['THUMB_MCP']], frame[LM['THUMB_IP']])
                    d2 = math.dist(frame[LM['THUMB_IP']],  frame[LM['THUMB_TIP']])
                    factors.append(d1 + d2)
                continue

            # PALM
            if self.task_norm_strategy == 'PALMSIZE':
                if has_idxs(frame,
                            LM['WRIST'],
                            LM['INDEX_FINGER_MCP'], LM['MIDDLE_FINGER_MCP'],
                            LM['RING_FINGER_MCP'], LM['PINKY_MCP']):
                    d1 = math.dist(frame[LM['WRIST']], frame[LM['INDEX_FINGER_MCP']])
                    d2 = math.dist(frame[LM['WRIST']], frame[LM['MIDDLE_FINGER_MCP']])
                    d3 = math.dist(frame[LM['WRIST']], frame[LM['RING_FINGER_MCP']])
                    d4 = math.dist(frame[LM['WRIST']], frame[LM['PINKY_MCP']])
                    factors.append((d1 + d2 + d3 + d4) / 4)
                continue
            
            # MAX AMPLITUDE
            if self.task_norm_strategy == 'MAXAMPLITUDE':
                if has_idxs(frame, LM['THUMB_TIP'], LM['INDEX_FINGER_TIP']):
                    dist_val = math.dist(frame[LM['THUMB_TIP']], frame[LM['INDEX_FINGER_TIP']])
                    factors.append(dist_val)
                continue

            # DEFAULTS TO INDEX
            if has_idxs(frame,
                        LM['INDEX_FINGER_MCP'], LM['INDEX_FINGER_PIP'],
                        LM['INDEX_FINGER_DIP'], LM['INDEX_FINGER_TIP']):
                d1 = math.dist(frame[LM['INDEX_FINGER_MCP']], frame[LM['INDEX_FINGER_PIP']])
                d2 = math.dist(frame[LM['INDEX_FINGER_PIP']], frame[LM['INDEX_FINGER_DIP']])
                d3 = math.dist(frame[LM['INDEX_FINGER_DIP']], frame[LM['INDEX_FINGER_TIP']])
                factors.append(d1 + d2 + d3)
            continue

        return max(factors) if factors else 1.0
# -------------------------------------------------------------
# --- END: Abstract methods definitions
# -------------------------------------------------------------