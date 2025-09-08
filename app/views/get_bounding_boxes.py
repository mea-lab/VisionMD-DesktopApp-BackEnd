from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os, uuid, time, json
from pymediainfo import MediaInfo
from hachoir.parser import createParser
from hachoir.metadata import extractMetadata
from app.analysis.detectors.yolo_detectors import yolo_tracker

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

@api_view(['GET'])
def get_bounding_boxes(request):

    # Get all variables set up and check if folder and file paths exist
    video_id = request.GET.get('id', None)
    if not video_id:
        return Response("Video project id not provided.", status=400)

    folder_path = os.path.join(settings.MEDIA_ROOT, "video_uploads")
    project_folder_path = os.path.join(folder_path, video_id)
    if not os.path.isdir(project_folder_path):
        return Response("Video project folder does not exist.", status=400)

    metadata_file_path = os.path.join(project_folder_path, "metadata.json")
    if not os.path.isfile(metadata_file_path):
        return Response("Video project metadata does not exist.", status=400)
    
    # Get video path
    metadata_dict = {}
    with open(metadata_file_path, 'r', encoding='utf-8') as f:
        metadata_dict = (json.load(f))['metadata']
    video_path = os.path.join(project_folder_path, metadata_dict['video_name'])
    if not os.path.isfile(video_path):
        return Response("Video file in project folder does not exist.", status=400)

    # Do YOLO stuff here
    try:
        print("Analysis started")
        start_time = time.time()

        rotation = get_rotation(video_path)

        # 1) Build path to YOLO model
        current_dir = os.path.dirname(__file__)
        pathtomodel = os.path.join(current_dir, '../analysis/models/yolov8n.pt')

        # 2) Run YOLO-based tracker
        result = yolo_tracker(video_path, rotation, pathtomodel, device='')

        print("Analysis done in %s seconds" % (time.time() - start_time))

        # 3) Dump bounding boxes to json
        bounding_boxes_path = os.path.join(project_folder_path, 'boundingBoxes.json')
        bounding_boxes_wrapped = {
            "boundingBoxes": result['boundingBoxes']
        }

        with open(bounding_boxes_path, 'w') as jf:
            json.dump(bounding_boxes_wrapped, jf, indent=4)
        
    except Exception as e:
        os.remove(bounding_boxes_path) if os.path.exists(bounding_boxes_path) else None
        result = {'Error': str(e)}
        return Response(result, status=400)


    return Response(result)