# tasks/base_task.py

import cv2
import mediapipe as mp
import numpy as np
import math
import os, uuid, time, json, traceback
from django.core.files.storage import FileSystemStorage
from abc import ABC, abstractmethod
from .base_task import BaseTask
from django.conf import settings
from rest_framework.response import Response

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from ultralytics import YOLO
import tensorflow as tf
import tensorflow_hub as tfhub
from tensorflow.python.eager.context import context
import torch
from scipy.signal import butter, filtfilt
from scipy.signal import periodogram
import numpy as np
import tensorrt as trt
import os

class HandTremorRightTask(BaseTask):
    """
    Base class for all tasks (hand movement, finger tap, leg agility, toe tapping, etc.)
    Each concrete subclass must implement these abstract methods for retrieving
    & processing landmarks.
    """

    # ------------------------------------------------------------------
    # --- START: Abstract properties to be implemented by subclasses ---
    # ------------------------------------------------------------------
    LANDMARKS = {
        "THUMB": 0,
        "INDEX": 1,
        "MIDDLE": 2,
        "RING": 3,
        "PINKY": 4,
    }

    video_id = None
    file_path = None
    file_name = None
    task_name = None

    fps = None
    start_time = None
    start_frame_idx = None
    end_time = None
    end_frame_idx = None

    focal_length = None
    height_cm = None

    original_bounding_box = None
    enlarged_bounding_box = None
    subject_bounding_boxes = None

    _modelHandLandmarkNano = None
    _modelMeTrabs = None
    _detectorFaceLandmarker = None
    # ----------------------------------------------------------------
    # --- END: Abstract properties to be implemented by subclasses ---
    # ----------------------------------------------------------------





    # ---------------------------------------------------------------
    # --- START: Abstract methods to be implemented by subclasses ---
    # ---------------------------------------------------------------
    def api_response(self, request):
        """
        Function that handles the api response for each task
        """
        try:
            # 1) Getting video parameters from request
            self.prepare_video_parameters(request)

            # 2) Instatite clasas detectors 
            self.get_detector()
            print("Passed detector getting")

            pixel_to_mm_conversion_factor = self.calculate_pixel_conversion()
            print("Passed pixel conversion", pixel_to_mm_conversion_factor)

            landmarks = self.extract_landmarks()
            tf.keras.backend.clear_session()
            context().clear_kernel_cache()
            print("Passed landmark extraction")



            tremorSignal_Vertical_mm, tremorSignal_Horizontal_mm = self.calculate_signal(landmarks, pixel_to_mm_conversion_factor)
            print("Passed signal calculation")

            tremorAmplitude_Vertical = np.ptp(tremorSignal_Vertical_mm)  # peak-to-peak amplitude in mm
            tremorPrincipalFrequency_Vertical, _ = self.principal_frequency(tremorSignal_Vertical_mm, fs=self.fps)

            tremorAmplitude_Horizontal = np.ptp(tremorSignal_Horizontal_mm)  # peak-to-peak amplitude in mm
            tremorPrincipalFrequency_Horizontal, _ = self.principal_frequency(tremorSignal_Horizontal_mm, fs=self.fps)
            # Combine vertical and horizontal tremor signals
            print(f"Tremor Amplitude (mm): {tremorAmplitude_Vertical}")
            print(f"Tremor Principal Frequency (Hz): {tremorPrincipalFrequency_Vertical}")
            print(f"Tremor Amplitude (mm): {tremorAmplitude_Horizontal}")
            print(f"Tremor Principal Frequency (Hz): {tremorPrincipalFrequency_Horizontal}")

            #  Build up response to API call
            response = {}
            response['File name'] = self.file_name
            response['Task name'] = self.task_name
            response['Tremor Vertical Amplitude (mm)'] = tremorAmplitude_Vertical
            response['Tremor Vertical Principal Frequency (Hz)'] = tremorPrincipalFrequency_Vertical
            response['Tremor Horizontal Amplitude (mm)'] = tremorAmplitude_Horizontal
            response['Tremor Horiztonal Principal Frequency (Hz)'] = tremorPrincipalFrequency_Horizontal

            response['signals'] = {
                "Tremor Vertical (mm)": tremorSignal_Vertical_mm.tolist(), 
                "Tremor Horizontal (mm)": tremorSignal_Horizontal_mm.tolist(),
            }

            
            landmarks_np = np.array(landmarks)
            middle = landmarks_np[:, [1, 2, 3], :]
            response['landMarks'] = [ arr.tolist() for arr in middle ]
        except Exception as e:
            return Response(f"Error with hand tremor right analysis: {str(e)}", status=500)
        finally:
            if hasattr(self, "video") and self.video is not None:
                self.video.release()
            tf.keras.backend.clear_session()
            context().clear_kernel_cache()

        return response

    
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
        # Check if video id, json_data, video folder and video metadata file all exist
        video_id = request.GET.get('id', None)
        if not video_id:
            raise Exception("Video project id not provided.")
        
        if 'json_data' not in request.POST:
            raise Exception("Missing 'json_data' in POST data")
        json_raw = request.POST['json_data']
        
        if not json_raw:
            raise Exception("Empty 'json_data' in POST data")
        json_data = json.loads(json_raw)
        
        folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads", video_id)
        if not os.path.isdir(folder_path):
            raise Exception("Video project folder does not exist.")

        metadata_path = os.path.join(folder_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise Exception("Metadata file for video does not exist.")
        
        with open(metadata_path, 'r', encoding='utf-8') as f:
            try:
                metadata = json.load(f)
            except json.JSONDecodeError:
                raise Exception(f"Metadata file '{metadata_path}' cannot be decoded.")

                    
        #Getting video attributes
        file_name = metadata["metadata"]["video_name"]
        file_path = os.path.join(settings.MEDIA_ROOT, "video_uploads", video_id, file_name)
        task_name = f"{json_data['task_name']}_{json_data['id']}"
        video = cv2.VideoCapture(file_path)
        video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video.get(cv2.CAP_PROP_FPS)
        start_time = json_data['start_time']
        end_time = json_data['end_time']
        start_frame_idx = math.floor(fps * start_time)
        end_frame_idx   = math.ceil(fps * end_time)
        original_bounding_box = json_data['boundingBox']
        subject_bounding_boxes = [box for box in json_data['subject_bounding_boxes'] if start_frame_idx <= box['frameNumber'] <= end_frame_idx]
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
        height_cm = int(json_data.get('height')) if json_data.get('height') else None


        # Getting camera properties
        field_of_view = int(json_data.get('field_of_view')) if json_data.get('field_of_view') else None
        sensor_height = int(json_data.get('sensor_height')) if json_data.get('sensor_height') else None
        sensor_width = int(json_data.get('sensor_width')) if json_data.get('sensor_width') else None
        focal_length = int(json_data.get('focal_length')) if json_data.get('focal_length') else None
        intrinsic_matrix = np.array(json_data.get('intrinsic_matrix')) if json_data.get('intrinsic_matrix') else None
        extrinsic_matrix = np.array(json_data.get('extrinsic_matrix')) if json_data.get('extrinsic_matrix') else None

        # focal length [pixels] = focal length [mm] / sensor pixel size [µm/pixels]

        if(sensor_height != None and sensor_width != None and focal_length != None and intrinsic_matrix == None):
            fx = focal_length / sensor_width * 1000
            cx = video_width / 2
            fy = focal_length / sensor_height * 1000
            cy = video_height / 2
            intrinsic_matrix = [
                [fx, 0,  cx],
                [0,  fy, cy],
                [0,  0,   0],
            ]


        if ( abs(len(subject_bounding_boxes) - (end_frame_idx - start_frame_idx + 1)) > 1 ):
            print("Number of frames", end_frame_idx - start_frame_idx)
            print("Len of subject bounding boxes", len(subject_bounding_boxes))
            raise Exception("Subject bounding boxes not found in all frames of the task. The chosen subject may not be correct.")

        if(height_cm == None):
            raise Exception("Invalid or missing height.")


        #Set all necessary class attributes
        self.file_name = file_name
        self.file_path = file_path
        self.task_name = task_name
        self.video = video
        self.fps = fps
        self.start_time = start_time
        self.end_time = end_time
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box
        self.subject_bounding_boxes = subject_bounding_boxes
        self.height_cm = height_cm

        self.field_of_view = field_of_view
        self.sensor_height = sensor_height
        self.sensor_width = sensor_width
        self.focal_length = focal_length
        self.intrinsic_matrix = intrinsic_matrix
        self.extrinsic_matrix = extrinsic_matrix
        
        return {
            "video": video,
            "file_name": file_name,
            "file_path": file_path,
            "original_bounding_box": original_bounding_box,
            "enlarged_bounding_box": enlarged_bounding_box,
            "start_time": start_time,
            "end_time": end_time,
            "start_frame_idx": start_frame_idx,
            "end_frame_idx": end_frame_idx,
            "focal_length": focal_length,
            "height_cm": height_cm,
        }


    def get_detector(self) -> object:
        """
        Getter for the detector used by the task.

        Returns an instance of the detector using the detectors classes
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")

            #Hand Landmark Detection Model - Nano
            nano_model_path = os.path.join(settings.BASE_DIR, 'app', 'analysis', 'models', 'best_hand_landmark_Nano.engine' )
            self._modelHandLandmarkNano = YOLO(nano_model_path, task='pose')

            #metrabs model for 3d pose estimation
            metrabs_model_path = os.path.join(settings.BASE_DIR, 'app', 'analysis', 'models', 'metrabs_eff2s_y4' )
            self._modelMeTrabs = tfhub.load(metrabs_model_path)       

        else:
            device = torch.device("cpu")

            #Hand Landmark Detection Model - Nano
            nano_model_path = os.path.join(settings.BASE_DIR, 'app', 'analysis', 'models', 'best_hand_landmark_Nano.pt' )
            self._modelHandLandmarkNano = YOLO(nano_model_path, task='pose')

            #metrabs model for 3d pose estimation 
            metrabs_model_path = os.path.join(settings.BASE_DIR, 'app', 'analysis', 'models', 'metrabs_eff2s_y4' )
            self._modelMeTrabs = tfhub.load(metrabs_model_path)      


        # load mediapipe hand landmark model
        #face landmark detection model for Iris detection
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        #define path to the face landmarker model
        face_model_path = os.path.join(settings.BASE_DIR, 'app', 'analysis', 'models', 'face_landmarker.task')
        FaceLandmarker = mp.tasks.vision.FaceLandmarker
        FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions

        # Create a face landmarker instance with the video mode:
        optionsFaceLandmarker = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=face_model_path),
            running_mode=VisionRunningMode.IMAGE,
            output_facial_transformation_matrixes= True,
            output_face_blendshapes= False,
            num_faces=1, 
            min_face_detection_confidence=0.1,
            min_face_presence_confidence=0.1,
            min_tracking_confidence=0.1)

        self._detectorFaceLandmarker = FaceLandmarker.create_from_options(optionsFaceLandmarker)


    def get_signal_analyzer(self) -> object:
        """
        Getter for the signal analyzer used by the task

        Returns an instance of the signal analyze using the analyzer classes
        """
        pass


    def calculate_signal(self, landmarks, pixel_to_mm_conversion_factor) -> list:
        """
        Given a set of display landmarks (one list per frame), return the raw 1D
        signal array.
        """

        tremorSignal_Vertical = self.bandpass_filter(np.array(landmarks)[:,[1,2,3],1].mean(axis=1),fs=self.fps)
        tremorSignal_Horizontal = self.bandpass_filter(np.array(landmarks)[:,[1,2,3],0].mean(axis=1),fs=self.fps)

        timeSignal = np.arange(len(tremorSignal_Vertical)) / self.fps  # time vector for the signal

        tremorSignal_Vertical_mm = tremorSignal_Vertical * pixel_to_mm_conversion_factor  # convert tremor signal from pixels to mm
        tremorSignal_Horizontal_mm = tremorSignal_Horizontal * pixel_to_mm_conversion_factor  # convert tremor signal from pixels to mm

        return tremorSignal_Vertical_mm, tremorSignal_Horizontal_mm

    def extract_landmarks(self) -> tuple:
        """
        Process video frames between start_frame and end_frame and extract hand landmarks
        """
        video_path = self.file_path
        Hand = 'left'
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_time = self.start_time
        end_time = self.end_time
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)
        boundingBox = self.enlarged_bounding_box



        keypoints_2d_left_NanoModel = []
        keypoints_2d_right_NanoModel = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if current_time > end_time:
                break

            # Rotate if needed
            if frame.shape[1] > frame.shape[0]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Crop to bounding box
            h, w = frame.shape[:2]
            x_min = max(0, boundingBox['x']-50)
            y_min = max(0, boundingBox['y']-50)
            x_max = min(w - 1, boundingBox['x'] + boundingBox['width'] + 50)
            y_max = min(h - 1, boundingBox['y'] + boundingBox['height'] + 50)
            croppedFrame = frame[y_min:y_max, x_min:x_max, :]

            # Run hand landmark detection
            resultsLandmarksNano = self._modelHandLandmarkNano.track(frame, verbose=False, conf=0.5)
            predictionsLandmarks = resultsLandmarksNano[0].keypoints.xy.cpu().numpy()
            land_1 = predictionsLandmarks[0]
            land_2 = predictionsLandmarks[1]
            if land_1[:,0].mean()> land_2[:,0].mean():
                keypoints_2d_left_NanoModel.append(land_1)
                keypoints_2d_right_NanoModel.append(land_2)

            else:
                keypoints_2d_left_NanoModel.append(land_2)
                keypoints_2d_right_NanoModel.append(land_1)

        cap.release()
        return keypoints_2d_left_NanoModel



    def calculate_normalization_factor(self, essential_landmarks) -> float:
        """
        Return a caluclated scalar factor used to normalize the raw 1D signal.
        """
        pass
    
    # -------------------------------------------------------------
    # --- END: Abstract methods to be implemented by subclasses ---
    # -------------------------------------------------------------





    # -------------------------------------------------------------
    # --- START: Custom helper methods definitions
    # -------------------------------------------------------------
    def calculate_pixel_conversion(self):
        video_path = self.file_path
        Hand = 'left'
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_time = self.start_time
        end_time = self.end_time
        cap.set(cv2.CAP_PROP_POS_MSEC, start_time * 1000)

        boundingBox = self.enlarged_bounding_box
        
        camera_args = {}
        if self.field_of_view is not None:
            camera_args["default_fov_degrees"] = int(self.field_of_view)
        if self.intrinsic_matrix is not None:
            camera_args["intrinsic_matrix"] = tf.convert_to_tensor(self.intrinsic_matrix.astype(np.float32))
        if self.extrinsic_matrix is not None:
            camera_args["extrinsic_matrix"] = tf.convert_to_tensor(self.extrinsic_matrix.astype(np.float32))
    
        left_iris_diameters = []
        right_iris_diameters = []
        depthFace = []
        depthRightHand = []
        depthLeftHand = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Rotate only if the image is wider than it is tall
            if frame.shape[1] > frame.shape[0]:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB) #change color space to RGB for MediaPipe processing

            # Convert the frame to RGB for MediaPipe processing
            h, w = frame.shape[:2]
            x_min = max(0, boundingBox['x']-50)
            y_min = max(0, boundingBox['y']-50)
            x_max = min(w - 1, boundingBox['x'] + boundingBox['width'] + 50)
            y_max = min(h - 1, boundingBox['y'] + boundingBox['height'] + 50)
            croppedFrame = frame[y_min:y_max, x_min:x_max,:]
            # Create a MediaPipe Image from the cropped frame
            imageCropped = mp.Image(image_format=mp.ImageFormat.SRGB, data=croppedFrame.astype(np.uint8))

            FaceLandmarkerResults = self._detectorFaceLandmarker.detect(imageCropped)

            # Get iris landmarks from FaceLandmarkerResults
            if FaceLandmarkerResults.face_landmarks:
                #compute the head pose
                R = FaceLandmarkerResults.facial_transformation_matrixes[0]

                sy = np.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

                singular = sy < 1e-6

                if  not singular :
                    x = np.arctan2(R[2,1] , R[2,2])
                    y = np.arctan2(-R[2,0], sy)
                    z = np.arctan2(R[1,0], R[0,0])
                else :
                    x = np.arctan2(-R[1,2], R[1,1])
                    y = np.arctan2(-R[2,0], sy)
                    z = 0

                if (np.degrees(x)<15) and (np.degrees(y)<15) and (np.degrees(z)<15):
                    # If the head pose is neutral, we can extract the iris landmarks

                    face_landmarks = FaceLandmarkerResults.face_landmarks[0]
                    # MediaPipe FaceLandmarker iris landmarks: left (468-473), right (473-478)
                    right_iris = face_landmarks[469:473] #[468:473]
                    left_iris = face_landmarks[474:478] # [473:478]
                    if len(left_iris) == 4:
                        # Convert normalized coordinates to pixel coordinates
                        h, w = imageCropped.numpy_view().shape[:2]
                        x0, y0 = int(left_iris[0].x * w), int(left_iris[0].y * h)
                        x1, y1 = int(left_iris[1].x * w), int(left_iris[1].y * h)
                        x2, y2 = int(left_iris[2].x * w), int(left_iris[2].y * h)
                        x3, y3 = int(left_iris[3].x * w), int(left_iris[3].y * h)
                        (x_C,y_C), lradius = cv2.minEnclosingCircle(np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]])) 
                        left_iris_diameters.append(lradius * 2)  # Diameter is twice the radius

                    if len(right_iris) == 4:
                        h, w = imageCropped.numpy_view().shape[:2]
                        x0, y0 = int(right_iris[0].x * w), int(right_iris[0].y * h)
                        x1, y1 = int(right_iris[1].x * w), int(right_iris[1].y * h)
                        x2, y2 = int(right_iris[2].x * w), int(right_iris[2].y * h)
                        x3, y3 = int(right_iris[3].x * w), int(right_iris[3].y * h)
                        (x_C,y_C), rradius = cv2.minEnclosingCircle(np.array([[x0, y0], [x1, y1], [x2, y2], [x3, y3]]))
                        right_iris_diameters.append(rradius * 2)  # Diameter is twice


                        #use MeTrabs to estimate the 3D pose
                        resultsMeTrabs = self._modelMeTrabs.detect_poses(frame, skeleton='mpi_inf_3dhp_17', **camera_args)
                    
                        if resultsMeTrabs:
                            pose = resultsMeTrabs['poses3d'][0].cpu().numpy()
                            # Convert the pose to a numpy array
                            depthFace.append(pose[16][-1])  # depth of head
                            depthRightHand.append(pose[4][-1])  # depth of right hand
                            depthLeftHand.append(pose[7][-1])  # depth of left hand

            if len(left_iris_diameters) > 10:
                break

        cap.release()


        if len(left_iris_diameters) > 0:
            ValidPose = True
            # Calculate the average iris diameter in pixels
            avg_left_iris_diameter_px = np.mean(left_iris_diameters)
            avg_right_iris_diameter_px = np.mean(right_iris_diameters)
            # Calculate the pixel-to-mm conversion factor
            if Hand == 'left':
                pixel_to_mm_conversion_factor = ( 11.7 / avg_left_iris_diameter_px) * (np.mean(depthFace) / (np.mean(depthFace) + (np.mean(depthFace) - np.mean(depthRightHand))))
            else:
                pixel_to_mm_conversion_factor = ( 11.7 / avg_right_iris_diameter_px) * (np.mean(depthFace) / (np.mean(depthFace) + (np.mean(depthFace) - np.mean(depthLeftHand))))
            print(f"Pixel-to-mm Conversion Factor: {pixel_to_mm_conversion_factor}")
        else:
            ValidPose = False
            pixel_to_mm_conversion_factor = 1
            raise Exception("No valid head pose detected. Cannot calculate iris diameter. \n Pixel-to-mm Conversion Factor: {pixel_to_mm_conversion_factor}")
        
        return pixel_to_mm_conversion_factor

    def bandpass_filter(self, signal, lowcut=2, highcut=10, fs=60):
        # Design Butterworth bandpass filter
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(N=2, Wn=[low, high], btype='band', analog=False)
        # Remove NaNs if any
        signal = np.asarray(signal)
        if np.isnan(signal).any():
            signal = np.nan_to_num(signal)
        return filtfilt(b, a, signal)

    # Function to estimate the principal frequency of a signal
    def principal_frequency(self, signal, fs):
        f, Pxx = periodogram(signal, fs=fs)
        idx = np.argmax(Pxx[1:]) + 1  # skip DC component
        return f[idx], np.sqrt(Pxx[idx])  # amplitude is sqrt of power at peak
    # -------------------------------------------------------------
    # --- END: Custom helper methods definitions
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
            landmark.x * (x2 - x1) +  x1 - ox1,
            landmark.y * (y2 - y1) +  y1 - oy1,
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
                lm.x * (x2 - x1) + x1 - ox1,
                lm.y * (y2 - y1) + y1 - oy1,
                lm.z
            ])
        return coords
    # ------------------------------------------------
    # --- END: Utility functions as static methods ---
    # ------------------------------------------------