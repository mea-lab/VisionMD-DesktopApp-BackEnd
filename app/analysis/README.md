# Analysis Module Documentation

This document provides an overview of the Analysis module, which is responsible for processing video data for movement analysis tasks related to medical/clinical assessments.

## Folder Structure

### detectors/

Contains implementations of different object and pose detection frameworks:

- **mediapipe_detectors.py**: Implementations using Google's MediaPipe framework for pose and hand landmark detection
- **yolo_detectors.py**: Implementations using YOLO (You Only Look Once) models for pose detection

Detectors provide a consistent interface for extracting landmarks/keypoints from video frames.

### models/

Contains trained model files used by the detectors:

- Hand and pose landmark models for MediaPipe
- YOLO NAS and YOLOv8 model files for pose detection

You can upload and use your own custom trained models in this directory to extend detection capabilities.

### signal_processors/

Contains utilities for processing and analyzing the signals extracted from videos:

- **signal_processor.py**: Classes and functions to process time series data of landmarks, including filtering, peak detection, and feature extraction

### tasks/

Contains implementations of specific analysis tasks:

- **base_task.py**: Abstract base class defining the interface for all analysis tasks
- Individual task implementations like finger_tap_left.py, hand_movement_right.py, etc.

Each task defines how to analyze a specific movement pattern. Adding a new file here automatically creates an API endpoint.

### video_uploads/

Temporary storage location for uploaded videos that will be processed.

## Adding a New Analysis Task

### Step 1: Create a new task class

Create a new Python file in the `tasks/` folder (e.g., `my_new_task.py`) and define a class that inherits from `BaseTask`:

```python
from .base_task import BaseTask
from app.analysis.detectors.mediapipe_detectors import create_mediapipe_hand_detector
from app.analysis.signal_processors.signal_processor import SignalAnalyzer

class MyNewTask(BaseTask):
    """
    My New Task:
      - Description of how the signal is calculated.
      - Description of normalization factor.
    """
    
    # Define your landmarks dictionary
    LANDMARKS = {
        "WRIST": 0,
        "THUMB_TIP": 4,
        # Add relevant landmarks here
    }
    
    # Implement required methods from BaseTask
    # The system will automatically create an API endpoint for this task
```

### Step 2: Choose or implement a detector

Either use an existing detector from `detectors/` or implement a new one if needed. You can use custom models by placing them in the `models/` directory.

```python
# Example of using a detector in your task
def get_detector(self):
    return create_mediapipe_hand_detector()
```

### Step 3: Process signals

Use the `SignalAnalyzer` class from `signal_processors/signal_processor.py` to process the landmark data:

```python
def process_signal(self, landmarks, normalization_factor):
    # Calculate your raw signal
    raw_signal = [self._calculate_distance(frame) for frame in landmarks]
    
    # The SignalAnalyzer will handle:
    # - Filtering, smoothing, and normalizing
    # - Peak and valley detection
    # - Feature extraction for metrics
    analyzer = SignalAnalyzer()
    return analyzer.analyze(raw_signal, normalization_factor)
```

### Step 4: No explicit registration needed

The system automatically detects your task file and creates corresponding API endpoints. The frontend can then access these endpoints based on the task name.

## Best Practices

1. Follow the separation of concerns pattern:
   - Detectors handle landmark extraction
   - Signal processors handle data cleaning and feature extraction
   - Task classes handle task-specific logic and reporting

2. Reuse existing components when possible

3. Document key parameters and assumptions in your task implementation

4. Include validation to ensure the video contains the expected movement

5. When adding custom models, ensure they follow the same input/output format as existing models

Each subdirectory will have its own detailed README with more specific information about its components and usage.