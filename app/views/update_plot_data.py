from rest_framework.decorators import api_view
from rest_framework.response import Response
import json, time
import numpy as np
import scipy.interpolate as interpolate


def updatePeaksAndValleys(inputJson):
    peaksData = inputJson['peaks_Data']
    peaksTime = inputJson['peaks_Time']
    valleysStartData =  inputJson['valleys_StartData'] 
    valleysStartTime =  inputJson['valleys_StartTime']
    valleysEndData =  inputJson['valleys_EndData']
    valleysEndTime =  inputJson['valleys_EndTime']
    velocityData = np.asarray(inputJson['velocity_Data'])
    velocityTime = np.asarray(inputJson['velocity_Time'])

    # Sort valleysStartTime and get the permutation indices
    sorted_indices = sorted(range(len(valleysStartTime)), key=lambda k: valleysStartTime[k])

    # Rearrange valleysStartTime
    valleysStartTime_sorted = sorted(valleysStartTime)

    # Rearrange valleysStartData based on sorted_indices
    valleysStartData_sorted = [valleysStartData[i] for i in sorted_indices]

    # Sort valleysEndTime and get the permutation indices
    sorted_indices_end = sorted(range(len(valleysEndTime)), key=lambda k: valleysEndTime[k])

    # Rearrange valleysEndTime
    valleysEndTime_sorted = sorted(valleysEndTime)

    # Rearrange valleysEndData based on sorted_indices_end
    valleysEndData_sorted = [valleysEndData[i] for i in sorted_indices_end]

    # Sort peaksTime and get the permutation indices
    sorted_indices_peaks = sorted(range(len(peaksTime)), key=lambda k: peaksTime[k])

    # Rearrange peaksTime
    peaksTime_sorted = sorted(peaksTime)

    # Rearrange peaksData based on sorted_indices_peaks
    peaksData_sorted = [peaksData[i] for i in sorted_indices_peaks]

    peaksTime = peaksTime_sorted
    peaksData = peaksData_sorted
    valleysEndTime = valleysEndTime_sorted
    valleysEndData = valleysEndData_sorted
    valleysStartTime = valleysStartTime_sorted
    valleysStartData = valleysStartData_sorted

    amplitude = []
    peakTime = []
    rmsVelocity = []
    speed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []
    cycleDuration = []

    for idx, item in enumerate(peaksData):
        # Height measures
        x1 = valleysStartTime[idx]
        y1 = valleysStartData[idx]

        x2 = valleysEndTime[idx]
        y2 = valleysEndData[idx]

        x = peaksTime[idx]
        y = peaksData[idx]

        f = interpolate.interp1d(np.array([x1, x2]), np.array([y1, y2]))

        amplitude.append(y - f(x))

        # Velocity

        idxStart = (np.abs(velocityTime - x1)).argmin()
        idxEnd = (np.abs(velocityTime - x2)).argmin()
        rmsVelocity.append(np.sqrt(np.mean(velocityData[idxStart:idxEnd] ** 2)))

        speed.append((y - f(x)) / ((valleysEndTime[idx] - valleysStartTime[idx])))
        averageOpeningSpeed.append((y - f(x)) / ((peaksTime[idx] - valleysStartTime[idx])))
        averageClosingSpeed.append((y - f(x)) / ((valleysEndTime[idx] - peaksTime[idx])))
        cycleDuration.append((valleysEndTime[idx] - valleysStartTime[idx]))

        # timming
        peakTime.append(peaksTime[idx] )

    meanAmplitude = np.mean(amplitude)
    stdAmplitude = np.std(amplitude)

    meanSpeed = np.mean(speed)
    stdSpeed = np.std(speed)

    meanRMSVelocity = np.mean(rmsVelocity)
    stdRMSVelocity = np.std(rmsVelocity)
    meanAverageOpeningSpeed = np.mean(averageOpeningSpeed)
    stdAverageOpeningSpeed = np.std(averageOpeningSpeed)
    meanAverageClosingSpeed = np.mean(averageClosingSpeed)
    stdAverageClosingSpeed = np.std(averageClosingSpeed)

    meanCycleDuration = np.mean(cycleDuration)
    stdCycleDuration = np.std(cycleDuration)
    rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    rate = len(valleysEndTime) / (valleysEndTime[-1] - valleysStartTime[0])

    cvAmplitude = stdAmplitude / meanAmplitude
    cvSpeed = stdSpeed / meanSpeed
    cvCycleDuration = stdCycleDuration / meanCycleDuration
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed

    numPeaksHalf = len(peaksData)//2
    rateDecay = (numPeaksHalf / (valleysEndTime[numPeaksHalf] - valleysStartTime[0])) / (numPeaksHalf / (valleysEndTime[-1] - valleysStartTime[numPeaksHalf]))

    amplitudeDecay = np.array(amplitude)[:len(amplitude)//2].mean() / np.array(amplitude)[len(amplitude)//2:].mean()
    velocityDecay = np.array(speed)[:len(speed)//2].mean() / np.array(speed)[len(speed)//2:].mean()

    radarTable = {
            "MeanAmplitude": meanAmplitude,
            "StdAmplitude": stdAmplitude,
            "MeanSpeed": meanSpeed,
            "StdSpeed": stdSpeed,
            "MeanRMSVelocity": meanRMSVelocity,
            "StdRMSVelocity": stdRMSVelocity,
            "MeanOpeningSpeed": meanAverageOpeningSpeed,
            "stdOpeningSpeed": stdAverageOpeningSpeed,
            "meanClosingSpeed": meanAverageClosingSpeed,
            "stdClosingSpeed": stdAverageClosingSpeed,
            "meanCycleDuration": meanCycleDuration,
            "stdCycleDuration": stdCycleDuration,
            "rangeCycleDuration": rangeCycleDuration,
            "rate": rate,
            "amplitudeDecay": amplitudeDecay,
            "velocityDecay": velocityDecay,
            "rateDecay": rateDecay,
            "cvAmplitude": cvAmplitude,
            "cvCycleDuration": cvCycleDuration,
            "cvSpeed": cvSpeed,
            "cvRMSVelocity" : cvRMSVelocity,
            "cvOpeningSpeed": cvAverageOpeningSpeed,
            "cvClosingSpeed": cvAverageClosingSpeed
        }
    
    return radarTable

@api_view(['POST'])
def updatePlotData(request):
    try:
        json_data = json.loads(request.POST['json_data'])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON data")

    try:
        print("Updating plot started")
        start_time = time.time()
        outputDict = updatePeaksAndValleys(json_data)
        print("Plot updated in %s seconds" % (time.time() - start_time))
        result = outputDict
    except Exception as e:
        print(f"Error in updatePlotData: {e}")
        result = {'error': str(e)}

    return Response(result)
