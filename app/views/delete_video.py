from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.conf import settings

import os
import shutil

@api_view(['DELETE'])
def delete_video(request):

    video_id = request.GET.get('id')

    if not video_id:
        return Response("Video project id not provided.", status=400)

    folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads")
    project_folder_path = os.path.join(folder_path, video_id)
    if not os.path.isdir(project_folder_path):
        return Response("Video project folder does not exist.", status=400)

    try:
        shutil.rmtree(project_folder_path)
    except Exception as e:
        return Response(f"Failed to delete project folder: {str(e)}", status=500)

    return Response(status=200)
