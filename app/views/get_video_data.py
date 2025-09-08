from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings
import os
import json

@api_view(['GET'])
def get_video_data(request):
    folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads")
    folder_id = request.GET.get('id', None)

    all_metadata = []

    # If the base folder does not exist, return an empty list
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path, exist_ok=True)

    # If a specific folder ID is provided
    if folder_id:
        subfolder_path = os.path.join(folder_path, folder_id)
        subfolder_data = {}
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.lower().endswith('.json'):
                    json_path = os.path.join(subfolder_path, filename)
                    try:
                        with open(json_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            subfolder_data.update(data)
                    except (IOError, json.JSONDecodeError):
                        return Response("Video project data cannot be decoded.", status=404)
        return Response(subfolder_data, status=200)

    # No specific ID provided; return all metadata
    for sub in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, sub)
        subfolder_data = {}
        if not os.path.isdir(subfolder_path):
            continue

        for filename in os.listdir(subfolder_path):
            if filename.lower().endswith('.json'):
                json_path = os.path.join(subfolder_path, filename)
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        subfolder_data.update(data)
                except (IOError, json.JSONDecodeError):
                    return Response("Video project data cannot be decoded.", status=404)
        all_metadata.append(subfolder_data)

    return Response(all_metadata, status=200)
