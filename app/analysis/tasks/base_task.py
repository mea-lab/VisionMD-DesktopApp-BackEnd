# tasks/base_task.py

import cv2
import mediapipe as mp
import numpy as np
import math
import os, uuid, time, json, traceback
from django.core.files.storage import FileSystemStorage
from abc import ABC, abstractmethod

class BaseTask(ABC):
    """
    Base class for all tasks (hand movement, finger tap, leg agility, toe tapping, etc.)
    Each concrete subclass must implement these abstract methods for retrieving
    & processing landmarks.
    """

    # ------------------------------------------------------------------
    # --- START: Abstract properties to be implemented by subclasses ---
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def LANDMARKS(self):
        """
        Should be a dictionary where each landmark
        constant (e.g., WRIST, THUMB_TIP) maps to its corresponding index.
        """
        pass

    @property
    @abstractmethod
    def video_id(self):
        pass

    @property
    @abstractmethod
    def file_path(self):
        pass

    @property
    @abstractmethod
    def file_name(self):
        pass

    @property
    @abstractmethod
    def task_name(self):
        pass
    
    @property
    @abstractmethod
    def fps(self):
        pass

    @property
    @abstractmethod
    def start_time(self):
        pass
    

    @property
    @abstractmethod
    def start_frame_idx(self):
        pass


    @property
    @abstractmethod
    def end_time(self):
        pass


    @property
    @abstractmethod
    def end_frame_idx(self):
        pass

    @property
    @abstractmethod
    def original_bounding_box(self):
        pass


    @property
    @abstractmethod
    def enlarged_bounding_box(self):
        pass
    # ----------------------------------------------------------------
    # --- END: Abstract properties to be implemented by subclasses ---
    # ----------------------------------------------------------------





    # ---------------------------------------------------------------
    # --- START: Abstract methods to be implemented by subclasses ---
    # ---------------------------------------------------------------
    @abstractmethod
    def api_response(self, request):
        """
        Function that handles the api response for each task
        """
        pass

    
    @abstractmethod
    def prepare_video_parameters(self, request):
        """
        Prepares video parameters from the HTTP request:
         - Parses JSON for bounding box and time codes.
         - Saves the uploaded video file.
         - Computes the expanded bounding box.
         - Determines FPS and start/end frame indices.
        Returns a dictionary of parameters. 
        MUST DEFINE ALL ABSTRACT PROPERTIES. 
        """
        pass


    @abstractmethod 
    def get_detector(self) -> object:
        """
        Getter for the detector used by the task.

        Returns an instance of the detector using the detectors classes
        """
        pass


    @abstractmethod
    def get_signal_analyzer(self) -> object:
        """
        Getter for the signal analyzer used by the task

        Returns an instance of the signal analyze using the analyzer classes
        """
        pass


    @abstractmethod
    def calculate_signal(self, essential_landmarks) -> list:
        """
        Given a set of display landmarks (one list per frame), return the raw 1D
        signal array.
        """
        pass

    @abstractmethod
    def extract_landmarks(self, detector) -> tuple:
        """
        Process video frames between start_frame and end_frame and extract hand landmarks 
        for the left hand from each frame.
        
        Returns:
            tuple: (essential_landmarks, all_landmarks)
            - essential_landmarks: a list of lists where each inner list contains the key landmark coordinates for that frame.
            - all_landmarks: a list of lists containing all the landmark coordinates for that frame.
        """
        pass


    @abstractmethod
    def calculate_normalization_factor(self, essential_landmarks) -> float:
        """
        Return a caluclated scalar factor used to normalize the raw 1D signal.
        """
        pass
    # -------------------------------------------------------------
    # --- END: Abstract methods to be implemented by subclasses ---
    # -------------------------------------------------------------





    # --------------------------------------------------
    # --- START: Utility functions as static methods ---
    # --------------------------------------------------
    @staticmethod
    def get_landmark_coords(landmark, enlarged_coords, original_coords):
        """
        Computes the (x, y) coordinates of a given landmark relative to the provided bounds.
        """
        x1, y1, x2, y2 = enlarged_coords
        ox1, oy1, ox2, oy2 = original_coords
        return [
            landmark.x * (x2 - x1) +  x1,
            landmark.y * (y2 - y1) +  y1,
        ]

    @staticmethod
    def get_all_landmarks_coord(landmarks, enlarged_coords, original_coords):
        """
        Processes a list of landmarks and returns their (x, y, z) coordinates relative
        to the provided bounds.
        """
        x1, y1, x2, y2 = enlarged_coords
        ox1, oy1, ox2, oy2 = original_coords
        coords = []
        for lm in landmarks:
            coords.append([
                lm.x * (x2 - x1) + x1,
                lm.y * (y2 - y1) + y1,
                lm.z
            ])
        return coords
    # ------------------------------------------------
    # --- END: Utility functions as static methods ---
    # ------------------------------------------------