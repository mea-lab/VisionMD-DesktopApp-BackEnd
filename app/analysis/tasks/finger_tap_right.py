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

from .base_task import BaseTask
from app.analysis.detectors.mp_hand_detector import HandDetector
from app.analysis.signal_analyzers.peakfinder_signal_analyzer import PeakfinderSignalAnalyzer

class FingerTapRightTask(BaseTask):

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
    
    # Properties are set via prepare_video_parameters.
    video_id = None
    file_name = None
    task_name = None
    video = None
    fps = None
    start_time = None
    start_frame_idx = None
    end_time = None
    end_frame_idx = None
    file_path = None
    original_bounding_box = None
    enlarged_bounding_box = None

# ------------------------------------------------------------------
# --- END: Abstract properties definitions
# ------------------------------------------------------------------





# -------------------------------------------------------------
# --- START: Abstract methods definitions
# -------------------------------------------------------------
    def api_response(self, request):
        try:
            # 1) Process video and define all abstract class parameters
            self.prepare_video_parameters(request)

            # 3) Get analyzer
            signal_analyzer = self.get_signal_analyzer()

            # 4) Extract landmarks using the defined detector
            result = FingerTapRightTask.extract_landmarks(
                video_path=self.file_path, 
                start_frame_idx=self.start_frame_idx,
                end_frame_idx=self.end_frame_idx, 
                fps=self.fps, 
                enlarged_bounding_box=self.enlarged_bounding_box, 
                original_bounding_box=self.original_bounding_box, 
                LANDMARKS=self.LANDMARKS
            )
            essential_landmarks, all_landmarks = result

            
            # 5) Calculate the signal using the land marks
            normalization_factor = self.calculate_normalization_factor(all_landmarks)
            print("Norm factor", normalization_factor)

            # 6) Calculate the  normalization factor using the land marks
            raw_signal = self.calculate_signal(essential_landmarks)
            
            # Get output from the signal analyzer
            results = signal_analyzer.analyze(
                normalization_factor=normalization_factor,
                raw_signal=raw_signal,
                start_time=self.start_time,
                end_time=self.end_time
            )
            
            # Structure output
            output = {}
            output['File name'] = self.file_name
            output['Task name'] = self.task_name
            output = output | results
            output["landMarks"] = essential_landmarks
            output["allLandMarks"] = all_landmarks
            output["normalization_factor"] = normalization_factor

        except Exception as e:
            return {'Error': str(e)}

        finally:
            if self.video and self.video.isOpened():
                self.video.release()

        return output
    
    def prepare_video_parameters(self, request):
        # Get all variables set up and check if folder and file paths exist
        video_id = request.GET.get('id', None)
        if not video_id:
            raise Exception("Video project id not provided.")
        
        try:
            json_data = json.loads(request.POST['json_data'])
        except (KeyError, json.JSONDecodeError):
            raise Exception("Invalid or missing 'json_data' in POST data")

        folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads")
        project_folder_path = os.path.join(folder_path, video_id)
        if not os.path.isdir(project_folder_path):
            raise Exception("Video project folder does not exist.")
        
        subfolder_path = os.path.join(folder_path, video_id)
        metadata = {}
        if os.path.isdir(subfolder_path):
            json_path = os.path.join(subfolder_path, "metadata.json")
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            except (IOError, json.JSONDecodeError):
                raise Exception("Warning: Video project data cannot be decoded.")                    
    
        file_name = metadata["metadata"]["video_name"]
        file_path = os.path.join(settings.MEDIA_ROOT, "video_uploads", video_id, file_name)
        task_name = f"{json_data['task_name']}_{json_data['id']}"
    
        video = cv2.VideoCapture(file_path)
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        original_bounding_box = json_data['boundingBox']
        start_time = json_data['start_time']
        end_time = json_data['end_time']
        norm_strategy = json_data['norm_strategy']
        fps = video.get(cv2.CAP_PROP_FPS)
        start_frame_idx = math.floor(fps * start_time)
        end_frame_idx = math.floor(fps * end_time)
        
        new_x = int(max(0, original_bounding_box['x'] - original_bounding_box['width'] * 0.125))
        new_y = int(max(0, original_bounding_box['y'] - original_bounding_box['height'] * 0.125))
        new_width = int(min(video_width - new_x, original_bounding_box['width'] * 1.25))
        new_height = int(min(video_height - new_y, original_bounding_box['height'] * 1.25))
        enlarged_bounding_box = {
            'x': new_x,
            'y': new_y,
            'width': new_width,
            'height': new_height
        }

        self.video = video
        self.video_id = video_id
        self.file_name = file_name
        self.task_name = task_name
        self.file_path = file_path
        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
        self.fps = fps
        self.norm_strategy = norm_strategy

        return {
            "video": video,
            "video_id": video_id,
            "file_name": file_name,
            "task_name": task_name,
            "file_path": file_path,
            "original_bounding_box": original_bounding_box,
            "enlarged_bounding_box": enlarged_bounding_box,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame_idx": start_frame_idx,
            "end_frame_idx": end_frame_idx,
            "fps": fps,
        }


    def get_detector(self) -> object:
        return HandDetector().get_detector()


    def get_signal_analyzer(self) -> object:
        return PeakfinderSignalAnalyzer()

    @staticmethod
    def extract_landmarks(video_path, start_frame_idx, end_frame_idx, fps, enlarged_bounding_box, original_bounding_box, LANDMARKS) -> tuple:
        detector = HandDetector().get_detector()
        essential_landmarks = []
        all_landmarks = []
        enlarged_coords = (
            enlarged_bounding_box['x'],
            enlarged_bounding_box['y'],
            enlarged_bounding_box['x'] + enlarged_bounding_box['width'],
            enlarged_bounding_box['y'] + enlarged_bounding_box['height']
        )
        original_coords = (
            original_bounding_box['x'],
            original_bounding_box['y'],
            original_bounding_box['x'] + original_bounding_box['width'],
            original_bounding_box['y'] + enlarged_bounding_box['height']
        )
        
        video = cv2.VideoCapture(video_path)
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        current_frame_idx = start_frame_idx

        while current_frame_idx < end_frame_idx:
            success, frame = video.read()
            if not success:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            x1, y1, x2, y2 = enlarged_coords
            image_data = rgb_frame[y1:y2, x1:x2, :].astype(np.uint8)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_data)
            timestamp = int(current_frame_idx / fps * 1000)
            detection_result = detector.detect_for_video(image, timestamp)
            
            # Look for the right hand
            hand_index = -1
            for idx, label in enumerate(detection_result.handedness):
                if label[0].category_name == "Right":
                    hand_index = idx
                    break

            if hand_index == -1 or not detection_result.hand_landmarks[hand_index]:
                essential_landmarks.append([])
                all_landmarks.append([])
            else:
                hand_landmarks = detection_result.hand_landmarks[hand_index]
                thumb = BaseTask.get_landmark_coords(hand_landmarks[LANDMARKS["THUMB_TIP"]], enlarged_coords, original_coords)
                index_finger = BaseTask.get_landmark_coords(hand_landmarks[LANDMARKS["INDEX_FINGER_TIP"]], enlarged_coords, original_coords)
                middle_finger = BaseTask.get_landmark_coords(hand_landmarks[LANDMARKS["MIDDLE_FINGER_TIP"]], enlarged_coords, original_coords)
                wrist = BaseTask.get_landmark_coords(hand_landmarks[LANDMARKS["WRIST"]], enlarged_coords, original_coords)
                essential = [thumb, index_finger]
                all_lms = BaseTask.get_all_landmarks_coord(hand_landmarks, enlarged_coords, original_coords)
                essential_landmarks.append(essential)
                all_landmarks.append(all_lms)
            
            current_frame_idx += 1

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
        LM = self.LANDMARKS
        factors = []

        def has_idxs(frame, *idxs):
            return all(i < len(frame) for i in idxs)

        for frame in landmarks:
            # THUMB
            if self.norm_strategy == 'THUMBSIZE':
                if has_idxs(frame, 
                            LM['THUMB_CMC'], LM['THUMB_MCP'], 
                            LM['THUMB_IP'], LM['THUMB_TIP']):
                    d1 = math.dist(frame[LM['THUMB_MCP']], frame[LM['THUMB_IP']])
                    d2 = math.dist(frame[LM['THUMB_IP']],  frame[LM['THUMB_TIP']])
                    factors.append(d1 + d2)
                continue

            # PALM
            if self.norm_strategy == 'PALMSIZE':
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
            if self.norm_strategy == 'MAXAMPLITUDE':
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