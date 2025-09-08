# Video Analysis Task Framework

This document provides a comprehensive guide on how to create your own video analysis tasks within the framework. The tasks are designed to analyze specific movements (like finger tapping, leg movements, etc.) from video data.

## Table of Contents
- [Overview](#overview)
- [Task Structure](#task-structure)
- [Creating Your Own Task](#creating-your-own-task)
    - [Step 1: Create a New Task File](#step-1-create-a-new-task-file)
    - [Step 2: Define the Task Class](#step-2-define-the-task-class)
    - [Step 3: Implement Required Properties](#step-3-implement-required-properties)
    - [Step 4: Implement Required Methods](#step-4-implement-required-methods)
- [Example Implementation](#example-implementation)
- [Testing Your Task](#testing-your-task)

## Overview

The task framework processes videos of specific movement tasks used in clinical assessments. Each task:

1. Takes a video as input along with timing and region-of-interest information
2. Extracts relevant body/hand landmarks using AI-powered detectors
3. Calculates a signal based on those landmarks
4. Analyzes the signal to extract meaningful metrics

The current implementation supports tasks like finger tapping, hand movements, leg agility, and toe tapping, with each task having separate files for left and right sides.

## Task Structure

All tasks inherit from the `BaseTask` class which defines the common structure and required methods. Each task needs to:

- Define which landmarks to track
- Extract those landmarks from each video frame
- Calculate a signal from the landmarks
- Determine a normalization factor for the signal

## Creating Your Own Task

### Step 1: Create a New Task File

Create a new Python file in the `app/analysis/tasks` directory, e.g., `my_new_task.py`. Follow the naming convention of other tasks.

```bash
touch app/analysis/tasks/my_new_task.py
```

### Step 2: Define the Task Class

Start by importing the necessary modules and defining your task class that inherits from `BaseTask`:

```python
import math
import cv2
import mediapipe as mp
import numpy as np
import os, uuid, time, json, traceback
from django.core.files.storage import FileSystemStorage

from .base_task import BaseTask
from app.analysis.detectors.mediapipe_detectors import create_mediapipe_pose_heavy  # or appropriate detector
from app.analysis.signal_processors.signal_processor import SignalAnalyzer

class MyNewTask(BaseTask):
        """
        My New Task:
            - Describe what the task is measuring and how it works
            - Explain which landmarks are tracked
            - Describe how the signal is calculated
            - Explain what the normalization factor represents
        """
```

### Step 3: Implement Required Properties

Define all required properties from the base class:

```python
        # Define landmarks used by your task
        LANDMARKS = {
                "NOSE": 0,
                "LEFT_SHOULDER": 11,
                "RIGHT_SHOULDER": 12,
                # Add all landmarks you need for your task
        }

        # These properties will be set when processing a video
        original_bounding_box = None
        enlarged_bounding_box = None
        video = None
        fps = None
        start_time = None
        start_frame_idx = None
        end_time = None
        end_frame_idx = None
```

### Step 4: Implement Required Methods

You must implement several methods for your task to function properly:

#### 4.1. The `api_response` Method

This is the main entry point for your task. It orchestrates the entire analysis process:

```python
def api_response(self, request):
        """
        Main entry point that handles the entire task processing flow.
        """
        try:
                # 1) Process video and set parameters
                self.prepare_video_parameters(request)
                
                # 2) Get appropriate detector
                detector = self.get_detector()
                
                # 3) Get signal analyzer
                signal_analyzer = self.get_signal_analyzer()
                
                # 4) Extract landmarks from video frames
                essential_landmarks, all_landmarks = self.extract_landmarks(detector)
                
                # 5) Calculate normalization factor
                normalization_factor = self.calculate_normalization_factor(essential_landmarks)
                
                # 6) Calculate raw signal
                raw_signal = self.calculate_signal(essential_landmarks)
                
                # 7) Analyze the signal
                output = signal_analyzer.analyze(
                        normalization_factor=normalization_factor,
                        raw_signal=raw_signal,
                        start_time=self.start_time,
                        end_time=self.end_time
                )
                
                # 8) Add additional data to output
                output["landMarks"] = essential_landmarks
                output["allLandMarks"] = all_landmarks
                output["normalization_factor"] = normalization_factor
                
                result = output
                
        except Exception as e:
                result = {'error': str(e)}
                traceback.print_exc()
        
        # 9) Cleanup
        if self.video:
                self.video.release()
        if hasattr(self, 'file_path') and os.path.exists(self.file_path):
                os.remove(self.file_path)
                
        return result
```

#### 4.2. Video Parameter Preparation

This method handles video loading and parameter extraction:

```python
def prepare_video_parameters(self, request):
        """
        Parses the request data, saves the video file, and sets up parameters.
        """
        # Parse the request data
        data = json.loads(request.body)
        video_base64 = data['video']
        bounding_box = data['boundingBox']
        start_time = data['startTime']  # in seconds
        end_time = data['endTime']      # in seconds
        
        # Save video to a temporary file
        file_name = f"{uuid.uuid4()}.mp4"
        file_path = os.path.join('temp', file_name)
        fs = FileSystemStorage(location='temp')
        
        # Convert base64 to video file
        import base64
        with open(file_path, 'wb') as f:
                f.write(base64.b64decode(video_base64.split(',')[1]))
        
        # Open the video file
        video = cv2.VideoCapture(file_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        
        # Calculate frame indices
        start_frame_idx = int(start_time * fps)
        end_frame_idx = int(end_time * fps)
        
        # Get original bounding box
        original_bounding_box = {
                'x': bounding_box['x'],
                'y': bounding_box['y'],
                'width': bounding_box['width'],
                'height': bounding_box['height']
        }
        
        # Calculate enlarged bounding box (for better detection)
        padding = 0.1  # 10% padding
        x = max(0, original_bounding_box['x'] - original_bounding_box['width'] * padding)
        y = max(0, original_bounding_box['y'] - original_bounding_box['height'] * padding)
        w = original_bounding_box['width'] * (1 + 2 * padding)
        h = original_bounding_box['height'] * (1 + 2 * padding)
        
        enlarged_bounding_box = {
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h)
        }
        
        # Set class properties
        self.video = video
        self.file_path = file_path
        self.original_bounding_box = original_bounding_box
        self.enlarged_bounding_box = enlarged_bounding_box
        self.start_time = start_time
        self.end_time = end_time
        self.fps = fps
        self.start_frame_idx = start_frame_idx
        self.end_frame_idx = end_frame_idx
```

#### 4.3. Detector and Signal Analyzer

Define which detector to use and create a signal analyzer:

```python
def get_detector(self):
        """
        Returns the appropriate detector for this task.
        Choose from:
        - create_mediapipe_pose_heavy() - for body pose detection
        - create_mediapipe_hand_detector() - for hand landmark detection
        """
        return create_mediapipe_pose_heavy()  # or appropriate detector for your task

def get_signal_analyzer(self):
        """
        Returns the signal analyzer object.
        """
        return SignalAnalyzer()
```

#### 4.4. The Landmark Extraction Method

This is one of the most important methods where you extract landmarks from each frame:

```python
def extract_landmarks(self, detector):
        """
        Processes each frame to extract the landmarks needed for analysis.
        
        Returns:
                tuple: (essential_landmarks, all_landmarks)
                        essential_landmarks contains just the key points needed for signal calculation
                        all_landmarks contains all detected points for visualization
        """
        video = self.video
        fps = self.fps
        start_frame_idx = self.start_frame_idx
        end_frame_idx = self.end_frame_idx

        # Get bounding box coordinates
        enlarged_bb = self.enlarged_bounding_box
        x1 = enlarged_bb['x']
        y1 = enlarged_bb['y']
        x2 = x1 + enlarged_bb['width']
        y2 = y1 + enlarged_bb['height']
        enlarged_coords = (x1, y1, x2, y2)

        essential_landmarks = []
        all_landmarks = []

        # Set video to start frame
        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        current_frame_idx = start_frame_idx

        while current_frame_idx < end_frame_idx:
                success, frame = video.read()
                if not success:
                        break

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cropped_frame = rgb_frame[y1:y2, x1:x2, :].astype(np.uint8)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cropped_frame)

                # Get timestamp for the current frame
                timestamp = int((current_frame_idx / fps) * 1000)
                detection_result = detector.detect_for_video(image, timestamp)

                # Different logic depending on whether it's pose or hand detection
                if hasattr(detection_result, 'pose_landmarks') and detection_result.pose_landmarks:
                        # For pose detection
                        landmarks = detection_result.pose_landmarks[0]
                        
                        # Extract essential landmarks for your task
                        # Example for a leg movement task:
                        left_shoulder = landmarks[self.LANDMARKS["LEFT_SHOULDER"]]
                        right_shoulder = landmarks[self.LANDMARKS["RIGHT_SHOULDER"]]
                        
                        # Calculate shoulder midpoint
                        shoulder_mid = [
                                ((left_shoulder.x + right_shoulder.x) / 2) * (x2 - x1),
                                ((left_shoulder.y + right_shoulder.y) / 2) * (y2 - y1)
                        ]
                        
                        # Add more landmark processing here specific to your task
                        # ...
                        
                        # Add processed landmarks to the lists
                        essential = [shoulder_mid]  # Add all your essential landmarks
                        all_lms = BaseTask.get_all_landmarks_coord(landmarks, enlarged_coords)
                        
                        essential_landmarks.append(essential)
                        all_landmarks.append(all_lms)
                elif hasattr(detection_result, 'hand_landmarks') and len(detection_result.hand_landmarks) > 0:
                        # For hand detection
                        # Find the hand of interest (left or right)
                        hand_index = -1
                        for idx, label in enumerate(detection_result.handedness):
                                if label[0].category_name == "Right":  # or "Left" based on your task
                                        hand_index = idx
                                        break
                                        
                        if hand_index != -1 and detection_result.hand_landmarks[hand_index]:
                                hand_landmarks = detection_result.hand_landmarks[hand_index]
                                
                                # Extract essential landmarks
                                # Example for finger tapping:
                                thumb = BaseTask.get_landmark_coords(hand_landmarks[self.LANDMARKS["THUMB_TIP"]], enlarged_coords)
                                index_finger = BaseTask.get_landmark_coords(hand_landmarks[self.LANDMARKS["INDEX_FINGER_TIP"]], enlarged_coords)
                                
                                essential = [thumb, index_finger]  # Add all essential landmarks
                                all_lms = BaseTask.get_all_landmarks_coord(hand_landmarks, enlarged_coords)
                                
                                essential_landmarks.append(essential)
                                all_landmarks.append(all_lms)
                        else:
                                # No hand detected
                                essential_landmarks.append([])
                                all_landmarks.append([])
                else:
                        # No landmarks detected
                        essential_landmarks.append([])
                        all_landmarks.append([])
                        
                current_frame_idx += 1

        return essential_landmarks, all_landmarks
```

#### 4.5. Signal Calculation

Define how to calculate your movement signal from the landmarks:

```python
def calculate_signal(self, essential_landmarks):
        """
        Calculates the signal from the extracted landmarks.
        For example, in finger tapping, this would be the distance between thumb and index finger.
        
        Args:
                essential_landmarks: List of essential landmarks for each frame
                
        Returns:
                list: The calculated signal values for each frame
        """
        signal = []
        prev_signal = 0  # Used when landmarks are missing
        
        for frame_lms in essential_landmarks:
                if len(frame_lms) < 2:  # Not enough landmarks detected
                        signal.append(prev_signal)
                        continue
                        
                # Example calculation for finger tapping (distance between landmarks)
                point1, point2 = frame_lms[0], frame_lms[1]
                distance = math.dist(point1, point2)
                
                prev_signal = distance
                signal.append(distance)
                
        return signal
```

#### 4.6. Normalization Factor Calculation

Define how to calculate a normalization factor for your signal:

```python
def calculate_normalization_factor(self, essential_landmarks):
        """
        Calculates a normalization factor to account for different video scales/distances.
        This could be the maximum distance between two stable landmarks, average distance, etc.
        
        Args:
                essential_landmarks: List of essential landmarks for each frame
                
        Returns:
                float: The normalization factor
        """
        # Example: Finding max distance between specific landmarks
        distances = []
        
        for frame_lms in essential_landmarks:
                if len(frame_lms) < 4:  # Need certain landmarks
                        continue
                        
                # Example: distance between wrist and middle finger
                point1, point3 = frame_lms[0], frame_lms[2]
                distance = math.dist(point1, point3)
                distances.append(distance)
                
        # Return max or average distance as normalization factor
        return float(np.max(distances)) if distances else 1.0
```

## Example Implementation

Here's a simplified example of a head nodding task that tracks the vertical movement of the nose:

```python
import math
import cv2
import mediapipe as mp
import numpy as np
import os, uuid, time, json, traceback
from django.core.files.storage import FileSystemStorage

from .base_task import BaseTask
from app.analysis.detectors.mediapipe_detectors import create_mediapipe_pose_heavy
from app.analysis.signal_processors.signal_processor import SignalAnalyzer

class HeadNoddingTask(BaseTask):
        """
        Head Nodding Task:
            - Tracks the vertical position of the nose relative to the eye midpoint.
            - The signal is the vertical distance between nose and eye midpoint.
            - Normalization factor is the average distance between eyes.
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
                # Add other landmarks as needed
        }
        
        # Required properties
        original_bounding_box = None
        enlarged_bounding_box = None
        video = None
        fps = None
        start_time = None
        start_frame_idx = None
        end_time = None
        end_frame_idx = None
        
        def api_response(self, request):
                try:
                        self.prepare_video_parameters(request)
                        detector = self.get_detector()
                        signal_analyzer = self.get_signal_analyzer()
                        essential_landmarks, all_landmarks = self.extract_landmarks(detector)
                        normalization_factor = self.calculate_normalization_factor(essential_landmarks)
                        raw_signal = self.calculate_signal(essential_landmarks)
                        
                        output = signal_analyzer.analyze(
                                normalization_factor=normalization_factor,
                                raw_signal=raw_signal,
                                start_time=self.start_time,
                                end_time=self.end_time
                        )
                        output["landMarks"] = essential_landmarks
                        output["allLandMarks"] = all_landmarks
                        output["normalization_factor"] = normalization_factor
                        result = output
                except Exception as e:
                        result = {'error': str(e)}
                        traceback.print_exc()
                
                if self.video:
                        self.video.release()
                if hasattr(self, 'file_path') and os.path.exists(self.file_path):
                        os.remove(self.file_path)
                return result
        
        def prepare_video_parameters(self, request):
                # Implementation similar to other tasks
                # ...
        
        def get_detector(self):
                return create_mediapipe_pose_heavy()
        
        def get_signal_analyzer(self):
                return SignalAnalyzer()
        
        def extract_landmarks(self, detector):
                # Similar to other pose detection tasks but focus on face landmarks
                # ...
                # For each frame extract:
                # - Nose position
                # - Eye midpoint position
                # ...
                return essential_landmarks, all_landmarks
        
        def calculate_signal(self, essential_landmarks):
                signal = []
                prev_signal = 0
                
                for frame_lms in essential_landmarks:
                        if len(frame_lms) < 2:
                                signal.append(prev_signal)
                                continue
                                
                        nose = frame_lms[0]
                        eye_midpoint = frame_lms[1]
                        
                        # Vertical distance (y-axis)
                        diff = eye_midpoint[1] - nose[1]
                        prev_signal = diff
                        signal.append(diff)
                        
                return signal
        
        def calculate_normalization_factor(self, essential_landmarks):
                distances = []
                
                for frame_lms in essential_landmarks:
                        if len(frame_lms) < 3:
                                continue
                                
                        left_eye = frame_lms[2]
                        right_eye = frame_lms[3]
                        
                        # Distance between eyes
                        d = math.dist(left_eye, right_eye)
                        distances.append(d)
                        
                return float(np.mean(distances)) if distances else 1.0
```

## Testing Your Task

Once you’ve added your new `<your_task_name>.py` file under **`analysis/tasks/`**, the system will automatically:

- **Discover** it (no edits to `__init__.py` required)  
- **Generate** a POST endpoint at  
  ```
  /api/tasks/<your_task_name>/
  ```

### How to test

1. **Start (or restart)** your Django server:  
   ```bash
   python manage.py runserver
   ```
2. **POST** to your task’s URL  
   ```
   POST http://<host>:<port>/api/tasks/<your_task_name>/
   ```
   - **Form field** `"json_data"` (stringified JSON):  
     ```json
     {
       "boundingBox": { "x": 120, "y": 80, "width": 400, "height": 600 },
       "start_time": 0.0,
       "end_time": 8.0
     }
     ```
   - **Form field** `"video"`: your `.mp4` file
3. **Inspect** the JSON response in your client—e.g.:  
   ```json
   {
     "peaks": [...],
     "frequency_hz": 1.6,
     "amplitude_norm": 0.83,
     "landMarks": [[...], ...],
     "normalization_factor": 212.5
   }
   ```

No manual URL or registration steps are needed—just drop in the file, restart, and your task is live!
