import json
import math
import os
from pathlib import Path

import pytest
from rest_framework.test import APIClient

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "VideoAnalysisToolBackend.settings")
pytestmark = pytest.mark.django_db(transaction=True)

@pytest.fixture(autouse=True)
def temp_media_root(settings, tmp_path):
    settings.MEDIA_ROOT = tmp_path / "media"
    settings.MEDIA_URL = "/media/"

def upload_video(client):
    video_path = Path(__file__).parent / "videos" / "hand_tremor_right_elbow_extended.mp4"

    with video_path.open("rb") as video_file:
        response = client.post(
            "/api/upload_video/",
            {"video": video_file},
            format="multipart",
        )

    assert response.status_code == 200
    return response

def build_subject_bounding_boxes(task_frames, chosen_subject_id):
    subject_bounding_boxes = []
    for frame in task_frames:
        selected_box = next((box for box in frame["data"] if box["id"] == chosen_subject_id), None)
        if selected_box is None:
            continue
        subject_bounding_boxes.append(
            {
                "frameNumber": frame["frameNumber"],
                "data": [selected_box],
            }
        )
    return subject_bounding_boxes

def test_upload_video(settings):
    client = APIClient()
    response = upload_video(client)
    video_id = response.data["metadata"]["id"]
    assert video_id is not None

def test_delete_video(settings):
    client = APIClient()
    response = upload_video(client)
    video_id = response.data["metadata"]["id"]
    assert video_id is not None

    response = client.delete(f"/api/delete_video/?id={video_id}")
    assert response.status_code == 200

def test_get_bounding_boxes(settings):
    client = APIClient()
    response = upload_video(client)
    video_id = response.data["metadata"]["id"]
    assert video_id is not None

    response = client.get(f"/api/get_bounding_boxes/?id={video_id}")

    assert response.status_code == 200
    assert "boundingBoxes" in response.data
    assert isinstance(response.data["boundingBoxes"], list)
    assert len(response.data["boundingBoxes"]) > 0

    first_frame = response.data["boundingBoxes"][0]
    assert set(first_frame.keys()) == {"frameNumber", "data"}
    assert isinstance(first_frame["frameNumber"], int)
    assert isinstance(first_frame["data"], list)

def test_hand_tremor_right_elbow_extended_analysis(settings):
    start_time = 1
    end_time = 4
    chosen_subject_id = 1
    client = APIClient()

    response = upload_video(client)
    video_id = response.data["metadata"]["id"]
    fps = response.data["metadata"]["fps"]
    assert video_id is not None
    assert fps is not None

    bbox_response = client.get(f"/api/get_bounding_boxes/?id={video_id}")
    assert bbox_response.status_code == 200
    assert "boundingBoxes" in bbox_response.data

    all_bounding_boxes = bbox_response.data["boundingBoxes"]
    start_frame = math.floor(fps * start_time)
    end_frame = math.ceil(fps * end_time)
    task_frames = [box for box in all_bounding_boxes if start_frame <= box["frameNumber"] <= end_frame]
    assert task_frames, "No bounding boxes found in selected hand_tremor_right_elbow_extended time range"

    subject_bounding_boxes = build_subject_bounding_boxes(task_frames, chosen_subject_id)
    assert subject_bounding_boxes, f"No per-frame bounding boxes found for subject id {chosen_subject_id}"

    json_data = {
        "id": 1,
        "task_name": "hand_tremor_right_elbow_extended",
        "boundingBox": subject_bounding_boxes[0]["data"][0],
        "subject_bounding_boxes": subject_bounding_boxes,
        "start_time": start_time,
        "end_time": end_time,
    }
    hand_tremor_right_elbow_extended_response = client.post(f"/api/hand_tremor_right_elbow_extended/?id={video_id}", {"json_data": json.dumps(json_data)}, format="multipart")
    assert hand_tremor_right_elbow_extended_response.status_code == 200
