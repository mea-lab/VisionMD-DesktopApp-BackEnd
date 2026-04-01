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


TASK_CASES = [
    {
        "name": "finger_tap_right",
        "video_filename": "finger_tap_right.mp4",
        "endpoint": "finger_tap_right",
        "task_name": "finger_tap_right",
        "start_time": 3,
        "end_time": 6,
        "extra_payload": {"norm_strategy": "INDEX"},
    },
    {
        "name": "finger_tap_left",
        "video_filename": "finger_tap_left.mp4",
        "endpoint": "finger_tap_left",
        "task_name": "finger_tap_left",
        "start_time": 3,
        "end_time": 6,
        "extra_payload": {"norm_strategy": "INDEX"},
    },
    {
        "name": "hand_tremor_right_elbow_extended",
        "video_filename": "hand_tremor_right_elbow_extended.mp4",
        "endpoint": "hand_tremor_right_elbow_extended",
        "task_name": "hand_tremor_right_elbow_extended",
        "start_time": 1,
        "end_time": 4,
    },
    {
        "name": "leg_agility_right",
        "video_filename": "leg_agility_right.mp4",
        "endpoint": "leg_agility_right",
        "task_name": "leg_agility_right",
        "start_time": 3,
        "end_time": 6,
    },
    {
        "name": "leg_agility_left",
        "video_filename": "leg_agility_left.mp4",
        "endpoint": "leg_agility_left",
        "task_name": "leg_agility_left",
        "start_time": 3,
        "end_time": 6,
    },
    {
        "name": "toe_tapping_left",
        "video_filename": "toe_tapping_left.mp4",
        "endpoint": "toe_tapping_left",
        "task_name": "toe_tapping_left",
        "start_time": 3,
        "end_time": 6,
    },
    {
        "name": "toe_tapping_right",
        "video_filename": "toe_tapping_right.mp4",
        "endpoint": "toe_tapping_right",
        "task_name": "toe_tapping_right",
        "start_time": 3,
        "end_time": 6,
    },
]


def upload_video(client, video_filename):
    video_path = Path(__file__).parent / "videos" / video_filename

    with video_path.open("rb") as video_file:
        response = client.post(
            "/api/upload_video/",
            {"video": video_file},
            format="multipart",
        )

    assert response.status_code == 200
    return response.data["metadata"]["id"], response.data["metadata"]["fps"]


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


def test_all_tasks_in_sequence(settings):
    client = APIClient()
    chosen_subject_id = 1
    uploaded = []

    # Step 1: upload all videos one after another.
    for case in TASK_CASES:
        video_id, fps = upload_video(client, case["video_filename"])
        assert video_id is not None
        assert fps is not None
        uploaded.append(
            {
                **case,
                "video_id": video_id,
                "fps": fps,
            }
        )

    # Step 2: run all analyses one after another.
    for case in uploaded:
        video_id = case["video_id"]
        fps = case["fps"]
        start_time = case["start_time"]
        end_time = case["end_time"]

        bbox_response = client.get(f"/api/get_bounding_boxes/?id={video_id}")
        assert bbox_response.status_code == 200
        assert "boundingBoxes" in bbox_response.data

        all_bounding_boxes = bbox_response.data["boundingBoxes"]
        assert isinstance(all_bounding_boxes, list)
        assert all_bounding_boxes, f"No bounding boxes found for {case['name']}"

        start_frame = math.floor(fps * start_time)
        end_frame = math.ceil(fps * end_time)
        task_frames = [
            box for box in all_bounding_boxes if start_frame <= box["frameNumber"] <= end_frame
        ]
        assert task_frames, f"No bounding boxes found in selected time range for {case['name']}"

        subject_bounding_boxes = build_subject_bounding_boxes(task_frames, chosen_subject_id)
        assert subject_bounding_boxes, (
            f"No per-frame bounding boxes found for subject id {chosen_subject_id} in {case['name']}"
        )

        json_data = {
            "id": 1,
            "task_name": case["task_name"],
            "boundingBox": subject_bounding_boxes[0]["data"][0],
            "subject_bounding_boxes": subject_bounding_boxes,
            "start_time": start_time,
            "end_time": end_time,
        }
        json_data.update(case.get("extra_payload", {}))

        analysis_response = client.post(
            f"/api/{case['endpoint']}/?id={video_id}",
            {"json_data": json.dumps(json_data)},
            format="multipart",
        )
        assert analysis_response.status_code == 200, f"Analysis failed for {case['name']}"

    # Step 3: delete all videos one after another.
    for case in uploaded:
        delete_response = client.delete(f"/api/delete_video/?id={case['video_id']}")
        assert delete_response.status_code == 200, f"Delete failed for {case['name']}"
