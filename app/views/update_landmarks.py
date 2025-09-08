from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.http import Http404
import importlib
import json


@api_view(['POST'])
def update_landmarks(request):
    """
    Given a JSON payload containing:
      - 'task_name'
      - 'landmarks' (i.e. display_landmarks)
      - 'fps', 'start_time', 'end_time'
      - 'allLandMarks'
      - 'normalization_factor'
    Re-run the final step of analysis (peak finding, stats, etc.)
    using the new pipeline structure.
    """
    try:
        json_data = json.loads(request.POST['json_data'])
    except (KeyError, json.JSONDecodeError):
        raise Http404("Invalid or missing 'json_data' in POST body")

    # Extract fields
    task_name = json_data.get('task_name')
    essential_landmarks = json_data.get('landmarks', [])
    start_time = float(json_data.get('start_time', 0))
    end_time = float(json_data.get('end_time', 0))
    all_landmarks = json_data.get('allLandMarks', [])
    normalization_factor = float(json_data.get('normalization_factor', 1.0))

    # 1) Get the Task
    file_name = task_name.lower().replace(" ", "_")
    class_name = task_name.title().replace(" ", "") + "Task"
    module_path = f"app.analysis.tasks.{file_name}"

    try:
        task_module = importlib.import_module(module_path)
    except ModuleNotFoundError:
        raise Http404(f"Couldn’t find task module '{module_path}'")

    try:
        TaskClass = getattr(task_module, class_name)
    except AttributeError:
        raise Http404(f"Module '{module_path}' has no class '{class_name}'")

    try:
        task = TaskClass()
        raw_signal = task.calculate_signal(essential_landmarks)
        signal_analyzer = task.get_signal_analyzer()
        output = signal_analyzer.analyze(
            normalization_factor=normalization_factor,
            raw_signal=raw_signal,
            start_time=start_time,
            end_time=end_time
        )

        output["landMarks"] = essential_landmarks
        output["allLandMarks"] = all_landmarks
        output["normalization_factor"] = normalization_factor
    except:
        raise Http404(f"Something going wrong")

    return Response(output)
