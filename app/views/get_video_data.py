from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import os
import json
import shutil
import cv2
import logging

logger = logging.getLogger(__name__)

def verify_project_folder(project_path):
    try:
        if not os.path.isdir(project_path):
            logger.error("Project path does not exist")
            return False
        
        files = os.listdir(project_path)
        if not files:
            shutil.rmtree(project_path, ignore_errors=True)
            logger.error("Project files do not exist")
            return False
        
        metadata = {}
        metadata_file_path = os.path.join(project_path, "metadata.json")
        if not os.path.isfile(metadata_file_path):
            logger.error("Metadata files does not exist")
            return False
        try:
            with open(metadata_file_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)["metadata"]
        except Exception as e:
            logger.error("Metadata file could not be opened")
            return False
        
        required_fields = [
            "id",
            "video_name",
            "stem_name",
            "file_type",
            "fps",
            "thumbnail_url",
            "video_url",
            "rotation",
            "last_edited",
        ]
        for field in required_fields:
            if field not in metadata or metadata[field] is None or metadata[field] == "":
                logger.error("metadata is bad")
                return False
        
        thumbnail_file_path = os.path.join(project_path, "thumbnail.jpg")
        if not os.path.isfile(thumbnail_file_path):
            logger.error("thumbnail_file_path is bad")
            return False
        
        video_file_path = os.path.join(project_path, metadata["video_name"])
        if not os.path.isfile(video_file_path):
            logger.error("video_file_path is bad")
            return False
        cap = cv2.VideoCapture(video_file_path)
        if not cap.isOpened():
            logger.error("opening video is bad")
            return False
        ret, frame = cap.read()
        if not ret:
            logger.error("reading frame is bad")
            return False
        cap.release()
    
    except Exception as e:
        logger.error(e)
        return False
    
    return True


@api_view(['GET'])
def get_video_data(request):
    all_projects_path = os.path.join(settings.MEDIA_ROOT, "video_uploads")
    folder_id = request.GET.get('id', None)

    if not os.path.isdir(all_projects_path):
        os.makedirs(all_projects_path, exist_ok=True)

    # If a specific folder ID is provided
    if folder_id:
        project_data = {}
        project_path = os.path.join(all_projects_path, folder_id)
        if not os.path.isdir(project_path):
            return Response("Video project does not exist", status=404)
        
        files = os.listdir(project_path)
        if not files:
            shutil.rmtree(project_path, ignore_errors=True)
            return Response("Video project data is missing", status=404)
        
        if not verify_project_folder(project_path):
            shutil.rmtree(project_path, ignore_errors=True)
            return Response("Video project data is malformed", status=404)
        
        for filename in files:
            try:
                if filename.lower().endswith('.json'):
                    json_path = os.path.join(project_path, filename)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        project_data.update(data)
            except Exception as e:
                logger.error(f"Error decoding json file {f} | {e}")
                shutil.rmtree(project_path, ignore_errors=True)
                return Response(f"Video project data corrupted | {e}", status=404)
            
        return Response(project_data, status=200)
        

    # No specific ID provided, return all data
    all_project_data = []
    for dir in os.listdir(all_projects_path):
        project_data = {}
        project_path = os.path.join(all_projects_path, dir)
        if not os.path.isdir(project_path):
            logger.error("Project path does not exist")
            continue
    
        files = os.listdir(project_path)
        if not files:
            logger.error("Project files does not exist")
            shutil.rmtree(project_path, ignore_errors=True)
            continue

        if not verify_project_folder(project_path):
            shutil.rmtree(project_path, ignore_errors=True)
            continue

        for filename in files:
            try:
                if filename.lower().endswith('.json'):
                    json_path = os.path.join(project_path, filename)
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        project_data.update(data)
            except Exception as e:
                logger.error(f"Error decoding json file {f} | {e}")
                shutil.rmtree(project_path, ignore_errors=True)
                project_data = {}
                continue
        
        if project_data:
            all_project_data.append(project_data)

    all_project_data.sort(
        key=lambda p: p.get("metadata", {}).get("last_edited", ""),
        reverse=True
    )
            
    return Response(all_project_data, status=200)
