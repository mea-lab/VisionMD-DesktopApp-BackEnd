# detectors/yolo_detectors.py

import os
import torch
import cv2
from ultralytics import YOLO

# Fallback for MPS on Mac
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

def create_yolo_detector(model_path="yolov8s.pt", device='cpu'):
    """
    Example function that creates a YOLO object 
    """
    # Choose device
    device = 'cuda' if torch.cuda.is_available() else device
    if torch.backends.mps.is_available():
        device = 'mps'

    return YOLO(model_path), device


def yolo_tracker(file_path, rotation, model_path="yolov8s.pt", device='cpu'):
    """
    Runs YOLOv8 tracking and returns bounding boxes (every 10 frames),
    remapping IDs to be consecutive.
    """
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else device
    if torch.backends.mps.is_available():
        device = 'mps'

    # Load YOLO model
    model = YOLO(model_path)

    cap = cv2.VideoCapture(file_path)
    boundingBoxes = []
    frameNumber = 0
    data = []  # store the last set of detections between frames

    # Dictionary to remap original YOLO IDs to consecutive ones
    id_map = {}
    next_id = 1

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        if rotation != 0:
            if rotation == 90:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            elif rotation == 180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            elif rotation == 270:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            else:
                raise ValueError("Rotation must be one of [0, 90, 180, 270]")

        if frameNumber % 10 == 0:
            # Run YOLOv8 tracking on this frame
            results = model.track(
                frame,
                persist=True,
                classes=[0],  # class 0 = person
                verbose=False,
                device=device
            )
            data = []

            if (len(results) > 0 and
                results[0].boxes is not None and
                results[0].boxes.id is not None):
                yolo_ids = results[0].boxes.id.cpu().numpy().astype(int)
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)

                for i in range(len(yolo_ids)):
                    original_id = int(yolo_ids[i])

                    # Remap to consecutive ID
                    if original_id not in id_map:
                        id_map[original_id] = next_id
                        next_id += 1

                    mapped_id = id_map[original_id]

                    temp = {
                        'id': mapped_id,
                        'x': int(boxes[i][0]),
                        'y': int(boxes[i][1]),
                        'width': int(boxes[i][2] - boxes[i][0]),
                        'height': int(boxes[i][3] - boxes[i][1]),
                        'Subject': False
                    }
                    data.append(temp)

            frameResults = {
                'frameNumber': frameNumber,
                'data': data
            }
            boundingBoxes.append(frameResults)
        else:
            # Repeat last known data
            frameResults = {
                'frameNumber': frameNumber,
                'data': data
            }
            boundingBoxes.append(frameResults)

        frameNumber += 1

    cap.release()
    return {'boundingBoxes': boundingBoxes}
