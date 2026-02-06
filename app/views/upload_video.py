from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os, time, cv2, json, base64
from datetime import datetime, timezone
from pymediainfo import MediaInfo
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
import subprocess
import shutil
import os, sys
import traceback

def get_ffmpeg_path():
    if getattr(sys, 'frozen', False):
        base_path = sys._MEIPASS
        return os.path.join(base_path, "ffmpeg")
    else:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            raise FileNotFoundError("ffmpeg not found in PATH")
        return ffmpeg_path
    
def is_vfr(input_path):
    ffmpeg_path = get_ffmpeg_path()
    ffmprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    print(f"Chosen ffmpeg binary path for ffmprobing video: {ffmprobe_path}")
    cmd = [
        ffmprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,avg_frame_rate",
        "-of", "json",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe VFR check failed:\n{result.stderr}")

    try:
        data = json.loads(result.stdout)
        if not data.get("streams"):
            raise RuntimeError("FFprobe VFR check returned no streams.")
        stream = data["streams"][0]
        r_frame = stream.get("r_frame_rate")
        avg_frame = stream.get("avg_frame_rate")
        return r_frame != avg_frame
    except Exception:
        raise RuntimeError("Failed to parse FFprobe VFR check output.")

def convert_to_cfr(input_path, fps):
    if not is_vfr(input_path):
        print("Video is CFR already, skipping conversion.")
        return

    base, ext = os.path.splitext(input_path)
    ffmpeg_path = get_ffmpeg_path()
    output_path = f"{base}_cfr{ext}"
    print(f"Chosen ffmpeg binary path for VFR to CFR conversion: {ffmpeg_path}")

    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    probe_cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=sample_aspect_ratio,display_aspect_ratio",
        "-of", "json",
        input_path,
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
    sar = None
    dar = None
    if probe_result.returncode == 0:
        try:
            probe_data = json.loads(probe_result.stdout)
            probe_stream = probe_data["streams"][0]
            sar = probe_stream.get("sample_aspect_ratio")
            dar = probe_stream.get("display_aspect_ratio")
        except Exception:
            sar = None
            dar = None

    vf_parts = [f"fps={fps}"]
    if sar and sar not in {"0:1", "N/A"}:
        vf_parts.append(f"setsar={sar}")
    if dar and dar not in {"0:1", "N/A"}:
        vf_parts.append(f"setdar={dar}")
    vf_value = ",".join(vf_parts)

    cmd = [
        f'{ffmpeg_path}', '-y',
        '-i', input_path,
        '-vf', vf_value,
        '-vsync', 'cfr',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'copy',
        output_path
    ]
    
    print(f"Subprocess.CREATE_NO_WINDOW value: {getattr(subprocess, 'CREATE_NO_WINDOW', 0)}")
    result = subprocess.run(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg CFR conversion failed:\n{result.stderr.decode('utf-8')}")
    
    os.remove(input_path)
    os.rename(output_path, input_path)

def convert_to_square_pixels(input_path):
    ffmpeg_path = get_ffmpeg_path()
    ffprobe_path = ffmpeg_path.replace("ffmpeg", "ffprobe")
    cmd = [
        ffprobe_path,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,sample_aspect_ratio,display_aspect_ratio",
        "-of", "json",
        input_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFprobe failed:\n{result.stderr}")

    data = json.loads(result.stdout)
    if not data.get("streams"):
        raise RuntimeError("FFprobe returned no video streams.")
    stream = data["streams"][0]
    width = stream.get("width")
    height = stream.get("height")
    sar = stream.get("sample_aspect_ratio")
    dar = stream.get("display_aspect_ratio")

    def parse_ratio(ratio):
        if not ratio or ratio in {"0:1", "N/A"}:
            return None
        try:
            num_str, den_str = ratio.split(":")
            num = float(num_str)
            den = float(den_str)
            if den == 0:
                return None
            return num / den
        except Exception:
            return None

    sar_ratio = parse_ratio(sar)
    dar_ratio = parse_ratio(dar)
    if width and height and (sar in {"1:1", "N/A", None}):
        pixel_ratio = width / height
        if dar_ratio is None or abs(pixel_ratio - dar_ratio) < 0.001:
            print("Video already has square pixels and SAR matches DAR, skipping normalization.")
            return input_path

    if width is None or height is None:
        raise RuntimeError("Could not read video dimensions for square pixel normalization.")

    if sar_ratio is None:
        sar_ratio = 1.0

    target_w = width
    target_h = height
    if sar_ratio != 1.0:
        target_w = int(round(width * sar_ratio))
    elif dar_ratio is not None:
        target_w = int(round(height * dar_ratio))
    else:
        print("No DAR reported and SAR is 1:1, skipping normalization.")
        return input_path

    if target_w % 2 != 0:
        target_w += 1

    output_dar = f"{target_w}:{target_h}"
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_square{ext}"
    cmd = [
        f"{ffmpeg_path}", "-y",
        "-i", input_path,
        "-vf", f"scale={target_w}:{target_h},setsar=1,setdar={output_dar}",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "copy",
        output_path
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg square pixel normalization failed:\n{result.stderr.decode('utf-8')}")

    os.remove(input_path)
    os.rename(output_path, input_path)
    return input_path

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
    try:
        print("Started processing video upload...")
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
        saved_name = fs.save(original_filename, video)
        saved_video_path = os.path.join(folder_path, saved_name)
        if not os.path.exists(saved_video_path):
            raise RuntimeError("Video file was not saved to disk.")
        stem_name, extension = os.path.splitext(original_filename)
        stem_name = stem_name
        file_type = extension.lstrip('.')

        cap = cv2.VideoCapture(saved_video_path)
        if not cap.isOpened():
            return Response("Cannot open video file after saving.", status=400)
        ret, frame = cap.read()
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to read a frame from the uploaded video.")
        if not fps or fps <= 0:
            raise RuntimeError(f"Invalid FPS detected: {fps}")

        convert_to_cfr(saved_video_path, fps)
        convert_to_square_pixels(saved_video_path)
        cap2 = cv2.VideoCapture(saved_video_path)
        ret, frame = cap2.read()
        fps = cap2.get(cv2.CAP_PROP_FPS)
        cap2.release()
        if not ret or frame is None:
            raise RuntimeError("Failed to read a frame after normalization.")
        if not fps or fps <= 0:
            raise RuntimeError(f"Invalid FPS after normalization: {fps}")

        rotation = get_rotation(saved_video_path)
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

        # Assemble metadata
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

        # Save metadata.json into the same folder
        metadata_path = os.path.join(folder_path, 'metadata.json')
        with open(metadata_path, 'w') as jf:
            json.dump(metadata_wrapped, jf, indent=4)

        # Return the metadata as JSON
        return Response(metadata_wrapped, status=200)
    except Exception as e:
        print(traceback.format_exc())
        return Response({"detail": f"{e}"}, status=500)
