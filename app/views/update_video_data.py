from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os, time, cv2, json, base64
from datetime import datetime, timezone

@api_view(['POST'])
def update_video_data(request):

    # Get all variables set up and check if folder and file paths exist
    video_id = request.GET.get('id', None)
    folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads")
    project_folder_path = os.path.join(folder_path, video_id)
    if not video_id:
        return Response("Video project id not provided.", status=400)
    if not os.path.isdir(project_folder_path):
        return Response("Video project folder does not exist.", status=400)
    
    file_name = request.GET.get('file_name', None)
    file_path = os.path.join(project_folder_path, file_name)
    if not file_name.endswith('.json'):
        return Response("Provided file name is not a JSON file.", status=400)
    
    old_data = None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            old_data = json.load(f)[(file_name.removesuffix('.json'))]
    except (IOError, json.JSONDecodeError):
        print("Video project data cannot be decoded.")
    


    try:
        incoming_data = request.data
        if old_data and isinstance(incoming_data, dict) and isinstance(old_data, dict):
            new_data = old_data | incoming_data
        else:
            new_data = incoming_data

        if file_name == "metadata.json":
            old_video_name = old_data['video_name']
            old_video_path = os.path.join(project_folder_path, old_video_name)

            if not os.path.isfile(old_video_path):
                return Response("Old video path is not file", status=400)
            
            new_video_name = incoming_data['video_name'] or old_video_name
            new_video_path = os.path.join(project_folder_path, new_video_name)

            os.rename(old_video_path, new_video_path)
        
        new_data_wrapped = {(file_name.removesuffix('.json')) : new_data}
    except Exception as e:
        return Response(f"Invalid request body: {str(e)}", status=400)

    data_path = os.path.join(project_folder_path, file_name)
    try:
        with open(data_path, 'w') as jf:
            json.dump(new_data_wrapped, jf, indent=4)
    except Exception as e:
        return Response(f"Error writing to file: {str(e)}", status=500)

    # Return success
    return Response(status=200)