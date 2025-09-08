from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os, time, cv2, json, base64
from datetime import datetime, timezone
from pymediainfo import MediaInfo
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata


def get_rotation(path):
    mi = MediaInfo.parse(path)
    for track in mi.tracks:
        if track.track_type == "Video" and getattr(track, 'rotation', None):
            try:
                return int(float(track.rotation))
            except ValueError:
                pass

    parser = createParser(path)
    if parser:
        meta = extractMetadata(parser)
        if meta and meta.has("rotation"):
            try:
                return int(meta.get("rotation").value)
            except Exception:
                pass

    return 0


@api_view(['POST'])
def upload_video(request):
    # Check that a file was uploaded under the video key
    if 'video' not in request.FILES:
        return Response(
            {"detail": "'video' field missing or no files uploaded."},
            status=400
        )

    # Determine the base directory where all project‐ID folders live
    upload_root = os.path.join(settings.MEDIA_ROOT, "video_uploads")
    os.makedirs(upload_root, exist_ok=True)

    # Build a set of existing project‐IDs (folders named as 8-digit strings)
    existing_ids = set()
    for name in os.listdir(upload_root):
        full_path = os.path.join(upload_root, name)
        if os.path.isdir(full_path) and name.isdigit() and len(name) == 8:
            existing_ids.add(int(name))

    # Find the lowest unused integer between 0 and 99,999,999
    new_id_int = None
    for candidate in range(0, 100_000_000):
        if candidate not in existing_ids:
            new_id_int = candidate
            break
    if new_id_int is None:
        return Response(
            {"detail": "All project IDs from 00000000 through 99999999 are already taken."},
            status=500
        )

    # Zero-pad to 8 digits
    new_id_str = f"{new_id_int:08d}"

    # Create a folder named by the new project ID
    folder_path = os.path.join(upload_root, new_id_str)
    os.makedirs(folder_path, exist_ok=True)

    # Save the uploaded video file into that folder
    video = request.FILES['video']
    original_filename = video.name
    fs = FileSystemStorage(location=folder_path)
    fs.save(original_filename, video)
    saved_video_path = os.path.join(folder_path, original_filename)
    stem_name, extension = os.path.splitext(original_filename)
    stem_name = stem_name
    file_type = extension.lstrip('.')

    rotation = get_rotation(saved_video_path)

    cap = cv2.VideoCapture(saved_video_path)
    if not cap.isOpened():
        return Response(
            {"detail": "Cannot open video file after saving."},
            status=400
        )

    ret, frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    rotation_map = {
        90: cv2.ROTATE_90_CLOCKWISE,
        180: cv2.ROTATE_180,
        270: cv2.ROTATE_90_COUNTERCLOCKWISE
    }
    if rotation in rotation_map:
        frame = cv2.rotate(frame, rotation_map[rotation])

    success, buffer = cv2.imencode('.jpg', frame)
    if not success:
        return Response({"detail": "Failed to encode thumbnail as JPEG."}, status=500)

    thumbnail_rel_name = "thumbnail.jpg"
    thumbnail_path = os.path.join(folder_path, thumbnail_rel_name)
    with open(thumbnail_path, 'wb') as f:
        f.write(buffer)


    video_url = os.path.join(settings.MEDIA_URL, "video_uploads", new_id_str, original_filename)
    thumbnail_url = os.path.join(settings.MEDIA_URL, "video_uploads", new_id_str, thumbnail_rel_name)

    # 10) Assemble metadata
    metadata = {
        "id": new_id_str,
        "video_name": video.name,
        "stem_name": stem_name,
        "file_type": file_type,
        "fps": fps,
        "thumbnail_url": thumbnail_url,
        "video_url": video_url,
        "rotation": rotation,
        "last_edited": datetime.now(timezone.utc).isoformat()
    }
    
    metadata_wrapped = {
        "metadata": metadata
    }

    # 11) Save metadata.json into the same folder
    metadata_path = os.path.join(folder_path, 'metadata.json')
    with open(metadata_path, 'w') as jf:
        json.dump(metadata_wrapped, jf, indent=4)

    # 12) Return the metadata as JSON
    return Response(metadata_wrapped, status=200)
