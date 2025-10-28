# analysis/core/signal_processor.py

import numpy as np
import scipy.signal as signal
import scipy.signal as signal
import scipy.interpolate as interpolate
from app.analysis.signal_analyzers.base_signal_analyzer import BaseSignalAnalyzer
from scipy.ndimage import median_filter
import math

WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20

class PeakfinderSignalAnalyzer(BaseSignalAnalyzer):
    """
    Handles signal processing:
      - normalization & upsampling
      - embedded peak finding
      - computation of all cycle metrics
      - returns a dict matching your required jsonFinal schema
    """

    def analyze(self, raw_signal, normalization_factor, start_time, end_time):
        # 1) Normalize
        signal_array = np.array(raw_signal, dtype=float)
        norm = normalization_factor if normalization_factor else 1.0
        signal_array /= norm

        # 2) Upsample to 60 Hz
        up_fps = 60
        duration = end_time - start_time
        n_samples = int(duration * up_fps)
        if n_samples < 2:
            n_samples = len(signal_array)
        up_sample_signal = signal.resample(signal_array, n_samples)
        features, distance, velocity, peaks = get_output(up_sample_signal)

        # 4) Build time array
        size = len(distance)
        line_time = [(i/size)*duration + start_time for i in range(size)]

        # 5) Extract peaks & valleys
        line_peaks = []
        line_peaks_time = []
        line_valleys = []
        line_valleys_time = []
        line_valleys_start = []
        line_valleys_start_time = []
        line_valleys_end = []
        line_valleys_end_time = []

        for pk in peaks:
            p = pk['peakIndex']
            vs = pk['openingValleyIndex']
            ve = pk['closingValleyIndex']

            # peak
            line_peaks.append(distance[p])
            line_peaks_time.append((p/size)*duration + start_time)
            line_valleys.append(distance[vs])
            line_valleys_time.append((vs/size)*duration + start_time)
            # opening valley
            line_valleys_start.append(distance[vs])
            line_valleys_start_time.append((vs/size)*duration + start_time)
            # closing valley
            line_valleys_end.append(distance[ve])
            line_valleys_end_time.append((ve/size)*duration + start_time)


        # Build final dict
        jsonFinal = {
            "linePlot": {
                "data": distance.tolist(),
                "time": line_time
            },
            "velocityPlot": {
                "data": velocity.tolist(),
                "time": line_time
            },
            "rawData": {
                "data": up_sample_signal.tolist(),
                "time": line_time
            },
            "peaks": {
                "data": line_peaks,
                "time": line_peaks_time
            },
            "valleys": {
                "data": line_valleys,
                "time": line_valleys_time
            },
            "valleys_start": {
                "data": line_valleys_start,
                "time": line_valleys_start_time
            },
            "valleys_end": {
                "data": line_valleys_end,
                "time": line_valleys_end_time
            },
            "radarTable": features,
        }

        return jsonFinal



# ---------------------------------------------------------------
# -------------------- Peak Finder & Helpers --------------------
# ---------------------------------------------------------------
def decayEstimation(Peaks, nSelectedPeaks=4):
    slope, b = np.polyfit(np.arange(len(Peaks)), Peaks, 1)
    if slope < 0:
        return slope
    else:
        return 0

def scaling(landmarks, scale='THUMBSIZE'):
    prevScale = []
    newScale = []

    for idx, landmark in enumerate(landmarks):
        if len(landmark) > 0:
            # Check if landmark has enough points
            required_indices = [WRIST, MIDDLE_FINGER_TIP]
            if all(idx < len(landmark) for idx in required_indices):
                wrist, middle_finger_tip = landmark[WRIST], landmark[MIDDLE_FINGER_TIP]
                try:
                    #original scaling method is distance between wrist and middle finger tip
                    #so compute original scaling methods and store it in prevScale
                    dist = math.dist(wrist, middle_finger_tip)
                    prevScale.append(dist)
                except Exception as e:
                    print(f"Error computing distance for frame {idx}: {e}")
                    continue  # Skip this frame

                #if new scaling mehthod is selected, compute the new scaling method
                if scale == 'THUMBSIZE':
                    required_indices = [THUMB_CMC, THUMB_TIP]
                    if all(idx < len(landmark) for idx in required_indices):
                        # thumb_base, thumb_tip = landmark[THUMB_CMC], landmark[THUMB_TIP]
                        try:
                            #compute the size of each phalange
                            dist1 = math.dist(landmark[THUMB_CMC], landmark[THUMB_MCP])
                            dist2 = math.dist(landmark[THUMB_MCP], landmark[THUMB_IP])
                            dist3 = math.dist(landmark[THUMB_IP], landmark[THUMB_TIP])
                            #compute the size of the thumb
                            thumb_size = dist1 + dist2 + dist3
                            newScale.append(thumb_size)
                            # dist = math.dist(thumb_base, thumb_tip)
                            # newScale.append(dist)
                        except Exception as e:
                            print(f"Error computing thumb size for frame {idx}: {e}")
                            newScale.append(prevScale[-1])  # Use previous scale
                    else:
                        # Handle missing thumb landmarks
                        print(f"Missing thumb landmarks in frame {idx}")
                        newScale.append(prevScale[-1])  # Use previous scale
                elif scale == 'INDEXSIZE':
                    required_indices = [INDEX_FINGER_MCP, INDEX_FINGER_PIP,INDEX_FINGER_DIP, INDEX_FINGER_TIP]
                    if all(idx < len(landmark) for idx in required_indices):
                        # index_base, index_tip = landmark[INDEX_FINGER_MCP], landmark[INDEX_FINGER_TIP]
                        try:
                            #compute the size of each phalange
                            dist1 = math.dist(landmark[INDEX_FINGER_MCP], landmark[INDEX_FINGER_PIP])
                            dist2 = math.dist(landmark[INDEX_FINGER_PIP], landmark[INDEX_FINGER_DIP])
                            dist3 = math.dist(landmark[INDEX_FINGER_DIP], landmark[INDEX_FINGER_TIP])
                            #compute the size of the index finger
                            index_size = dist1 + dist2 + dist3
                            newScale.append(index_size)
                            # dist = math.dist(index_base, index_tip)
                            # newScale.append(dist)
                        except Exception as e:
                            print(f"Error computing index finger size for frame {idx}: {e}")
                            newScale.append(prevScale[-1])  # Use previous scale
                    else:
                        # Handle missing index finger landmarks
                        print(f"Missing index finger landmarks in frame {idx}")
                        newScale.append(prevScale[-1])  # Use previous scale
                elif scale == 'PALMSIZE':
                    required_indices = [WRIST, INDEX_FINGER_MCP, MIDDLE_FINGER_MCP, RING_FINGER_MCP, PINKY_MCP]
                    if all(idx < len(landmark) for idx in required_indices):
                        try:
                            #compute the average distance from the wrist to each finger
                            dist1 = math.dist(landmark[INDEX_FINGER_MCP], landmark[WRIST])
                            dist2 = math.dist(landmark[MIDDLE_FINGER_MCP], landmark[WRIST])
                            dist3 = math.dist(landmark[RING_FINGER_MCP], landmark[WRIST])
                            dist4 = math.dist(landmark[PINKY_MCP], landmark[WRIST])
                            #compute the size of the palm
                            palm_size = (dist1 + dist2 + dist3 +dist4 )/4
                            newScale.append(palm_size)
                        except Exception as e:
                            print(f"Error computing palm size for frame {idx}: {e}")
                            newScale.append(prevScale[-1])
                    else:
                        # Handle missing index finger landmarks
                        print(f"Missing hand base landmarks in frame {idx}")
                        newScale.append(prevScale[-1])  # Use previous scale
                else:
                    newScale.append(prevScale[-1])
            else:
                # Not enough points in landmark, skip this frame
                print(f"Missing required landmarks in frame {idx}")
                continue
        else:
            print(f"Empty landmark data in frame {idx}")
            continue

    # Check if prevScale and newScale are not empty to avoid division by zero
    if len(prevScale) == 0 or len(newScale) == 0:
        # Handle case where scaling cannot be computed
        print("Scaling cannot be computed due to missing landmarks.")
        return 1, 'NOSCALING'  # Return scaling factor of 1 (no scaling)
    else:
        # Divide the new scaling method by the original scaling method
        # to get the scaling factor
        scalingFactor = np.max(median_filter(prevScale, 3)) / np.max(median_filter(newScale, 3))
        return scalingFactor, scale  # Return scaling factor and scaling method

def get_output(up_sample_signal):
    fs = 60
    distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity = peakFinder(
        up_sample_signal, fs=fs, minDistance=3, cutOffFrequency=7.5, prct=0.05
    )

    amplitude = []
    peakTime = []
    rmsVelocity = []
    speed = []
    averageOpeningSpeed = []
    averageClosingSpeed = []
    maxOpeningSpeed = []
    maxClosingSpeed = []
    cycleDuration = []
    pauseDuration = []
    hesitationsinMovement = []
    hesitationsinPause = []
   

    Npeaks = len(peaks)
    maxVelocity = np.max(velocity)

    for idx, peak in enumerate(peaks):

        #for some reason, the peakFinder function does not return the opening and closing Peak Index
        # so we need to fill the empty values with the peak index


        #Opening sequence
        x1_start = peak['openingValleyIndex']* (1 / fs)
        y1_start = distance[peak['openingValleyIndex']]
        x1_end = peak['openingPeakIndex']* (1 / fs)
        y1_end = distance[peak['openingPeakIndex']]
        #Closing sequence
        x2_start = peak['closingPeakIndex']* (1 / fs)
        y2_start = distance[peak['closingPeakIndex']]
        x2_end = peak['closingValleyIndex']* (1 / fs)
        y2_end = distance[peak['closingValleyIndex']]
        #peak
        x = peak['peakIndex'] * (1 / fs)
        y = distance[peak['peakIndex']]

        #To identify the heigh of the peak, we create a line between the two points corresponding to the opening and closing. Then we compute the distance between the peak and the line
        #This is a function that can be evaluated at any value of x between x1 and x2
        f = interpolate.interp1d(np.array([x1_start, x2_end]), np.array([y1_start, y2_end]))
        amplitude.append(np.abs(y - f(x)))

        # Root Mean Square Velocity between the opening and closing valleys
        rmsVelocity.append(np.sqrt(np.mean(velocity[peak['openingValleyIndex']:peak['closingValleyIndex']] ** 2)))
        # Speed is the distance between the peak and the line divided by the time between the opening and closing valleys
        speed.append(
            (y - f(x)) / ((x2_end - x1_start) )
        )
        #Opening Speed is the distance between the peak and the line divided by the time between the opening valley and the peak of the opening sequence
        averageOpeningSpeed.append(
            np.abs((y1_end - f(x1_end))) / ((x1_end - x1_start) )
        )
        #closing Speed
        averageClosingSpeed.append(
            np.abs((y2_start - f(x2_start))) / ((x2_end - x2_start) )
        )
        #max Opening Speed
        maxOpeningSpeed.append(
            np.max(velocity[peak['openingValleyIndex']:peak['openingPeakIndex']])
        )
        #max Closing Speed
        maxClosingSpeed.append(
            np.max(np.abs(velocity[peak['closingPeakIndex']:peak['closingValleyIndex']]))
        )
        # Cycle duration
        cycleDuration.append(
            ((peak['closingValleyIndex'] - peak['openingValleyIndex']) * (1 / fs) )
        )
        # Timing
        peakTime.append(peak['peakIndex'] * (1 / fs))

        #pause time 
        if idx < Npeaks - 1:
            pauseDuration.append((peaks[idx + 1]['openingValleyIndex'] - peak['closingValleyIndex']) * (1 / fs))
            if pauseDuration[-1] < 0:
                pauseDuration[-1] = 0

        # calculate hesitations in the openeing-closing movement sequences 
        abs_velocity_segment = np.abs(velocity[peak['openingValleyIndex']+1:peak['closingValleyIndex']-1])
        # Count the number of times the absolute value of velocity crosses the threshold defined by maxVelocity*0.25
        crossings = np.sum((abs_velocity_segment[:-1] < maxVelocity*0.25) & (abs_velocity_segment[1:] >= maxVelocity*0.25)) + \
                    np.sum((abs_velocity_segment[:-1] >= maxVelocity*0.25) & (abs_velocity_segment[1:] < maxVelocity*0.25))
        
        # If the number of crossings is greater than 4, a hesitation occurred
        if crossings > 4:
            hesitationsinMovement.append(1)
        else:
            hesitationsinMovement.append(0)


        # calculate hesitations in the during pauses
        if idx < Npeaks - 1:
            if pauseDuration[-1] > 0.2: #only consider pauses longer than 0.1 seconds 
                abs_velocity_segment = np.abs(velocity[peak['closingValleyIndex']:peaks[idx + 1]['openingValleyIndex']+1])
                # Count the number of times the absolute value of velocity crosses the threshold defined by maxVelocity*0.25
                crossings = np.sum((abs_velocity_segment[:-1] < maxVelocity*0.25) & (abs_velocity_segment[1:] >= maxVelocity*0.25)) + \
                            np.sum((abs_velocity_segment[:-1] >= maxVelocity*0.25) & (abs_velocity_segment[1:] < maxVelocity*0.25))
                
                # If the number of crossings is greater than 4, a hesitation occurred
                if crossings > 4:
                    hesitationsinPause.append(1)
                else:
                    hesitationsinPause.append(0)


    if len(amplitude) == 0:
        print("No peaks detected; cannot compute output parameters.")
        return None

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

    meanMaxOpeningSpeed = np.mean(maxOpeningSpeed)
    stdMaxOpeningSpeed = np.std(maxOpeningSpeed)

    meanMaxClosingSpeed = np.mean(maxClosingSpeed)
    stdMaxClosingSpeed = np.std(maxClosingSpeed)


    meanCycleDuration = np.mean(cycleDuration)
    stdCycleDuration = np.std(cycleDuration)

    meanPauseDuration = np.mean(pauseDuration)
    stdPauseDuration = np.std(pauseDuration)


    # Compute the number of pauses as the number of cycles with a duration greater than 2 times the mean cycle duration
    # and the number of pauses with a duration greater than 2 times the mean pause duration
    numPauses = 0 
    numPauses = sum(1 for duration in cycleDuration if duration > 2 * meanCycleDuration) + sum(1 for duration in pauseDuration if duration > 2 * meanPauseDuration)

    hesitations = np.sum(hesitationsinMovement) + np.sum(hesitationsinPause)

    # Compute the range of cycle duration
    if len(peakTime) > 1:
        rangeCycleDuration = np.max(np.diff(peakTime)) - np.min(np.diff(peakTime))
    else:
        rangeCycleDuration = 0

    #Compute the average frequency as the number of peaks divided by the time between the first and last peak
    frequency = len(peaks) / ((peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']) * (1 / fs))

    # Initialize decay variables
    rateDecay = np.nan
    amplitudeDecay = np.nan
    velocityDecay = np.nan

    # Check if there are enough peaks to compute decay parameters
    if len(peaks) >= 3:
        n = len(peaks) // 3 # Split the peaks into 3 parts
        if n == 0:
            n = 1  # Ensure at least one peak is selected

        earlyPeaks = peaks[:n]
        latePeaks = peaks[-n:]

        # Ensure earlyPeaks and latePeaks are not empty
        if earlyPeaks and latePeaks and len(earlyPeaks) > 0 and len(latePeaks) > 0:
            # Rate Decay
            earlyDuration = (earlyPeaks[-1]['closingValleyIndex'] - earlyPeaks[0]['openingValleyIndex']) * (1 / fs)
            lateDuration = (latePeaks[-1]['closingValleyIndex'] - latePeaks[0]['openingValleyIndex']) * (1 / fs)

            earlyRate = len(earlyPeaks) / earlyDuration if earlyDuration != 0 else np.nan
            lateRate = len(latePeaks) / lateDuration if lateDuration != 0 else np.nan

            rateDecay = earlyRate / lateRate if lateRate and lateRate != 0 else np.nan

            # Amplitude Decay
            earlyAmplitude = np.array(amplitude)[:n]
            lateAmplitude = np.array(amplitude)[-n:]
            if np.mean(lateAmplitude) != 0:
                amplitudeDecay = np.mean(earlyAmplitude) / np.mean(lateAmplitude)
            else:
                amplitudeDecay = np.nan

            # Velocity Decay
            earlySpeed = np.array(speed)[:n]
            lateSpeed = np.array(speed)[-n:]
            if np.mean(lateSpeed) != 0:
                velocityDecay = np.mean(earlySpeed) / np.mean(lateSpeed)
            else:
                velocityDecay = np.nan

    # Coefficient of Variation
    cvAmplitude = stdAmplitude / meanAmplitude if meanAmplitude != 0 else np.nan
    cvSpeed = stdSpeed / meanSpeed if meanSpeed != 0 else np.nan
    cvRMSVelocity = stdRMSVelocity / meanRMSVelocity if meanRMSVelocity != 0 else np.nan
    cvAverageOpeningSpeed = stdAverageOpeningSpeed / meanAverageOpeningSpeed if meanAverageOpeningSpeed != 0 else np.nan
    cvAverageClosingSpeed = stdAverageClosingSpeed / meanAverageClosingSpeed if meanAverageClosingSpeed != 0 else np.nan
    cvMaxOpeningSpeed = stdMaxOpeningSpeed / meanMaxOpeningSpeed if meanMaxOpeningSpeed != 0 else np.nan
    cvMaxClosingSpeed = stdMaxClosingSpeed / meanMaxClosingSpeed if meanMaxClosingSpeed != 0 else np.nan
    cvCycleDuration = stdCycleDuration / meanCycleDuration if meanCycleDuration != 0 else np.nan
    cvPauseDuration = stdPauseDuration / meanPauseDuration if meanPauseDuration != 0 else np.nan


    jsonFinal = {
        "MeanAmplitude": float(meanAmplitude),
        "StdAmplitude": float(stdAmplitude),
        "MeanSpeed": float(meanSpeed),
        "StdSpeed": float(stdSpeed),
        "MeanRMSVelocity": float(meanRMSVelocity),
        "StdRMSVelocity": float(stdRMSVelocity),
        "MeanOpeningSpeed": float(meanAverageOpeningSpeed),
        "StdOpeningSpeed": float(stdAverageOpeningSpeed),
        "MeanClosingSpeed": float(meanAverageClosingSpeed),
        "StdClosingSpeed": float(stdAverageClosingSpeed),
        "MeanMaxOpeningSpeed": float(meanMaxOpeningSpeed),
        "StdMaxOpeningSpeed": float(stdMaxOpeningSpeed),
        "MeanMaxClosingSpeed": float(meanMaxClosingSpeed),
        "StdMaxClosingSpeed": float(stdMaxClosingSpeed),
        "MeanCycleDuration": float(meanCycleDuration),
        "StdCycleDuration": float(stdCycleDuration),
        # "MeanPauseDuration": meanPauseDuration,
        # "StdPauseDuration": stdPauseDuration,
        "CVAmplitude": float(cvAmplitude),
        "CVSpeed": float(cvSpeed),
        "CVRMSVelocity": float(cvRMSVelocity),
        "CVOpeningSpeed": float(cvAverageOpeningSpeed),
        "CVClosingSpeed": float(cvAverageClosingSpeed),
        "CVMaxOpeningSpeed": float(cvMaxOpeningSpeed),
        "CVMaxClosingSpeed": float(cvMaxClosingSpeed),
        "CVCycleDuration": float(cvCycleDuration),
        # "CVPauseDuration": cvPauseDuration,
        "Frequency": float(frequency),
        "AmplitudeDecay": float(amplitudeDecay),
        "VelocityDecay": float(velocityDecay),
        # "RateDecay": rateDecay,
        "RangeCycleDuration": float(rangeCycleDuration),
        "NumberofPauses": float(numPauses),
        "numberofHesitations": float(hesitations),
    }
    jsonFinal = {k: float(np.nan_to_num(v, nan=0.0)) for k, v in jsonFinal.items()}

    return jsonFinal, distance, velocity, peaks



def compareNeighboursNegative(item1, item2, distance, minDistance=5):
    # case 1 -> item1 peak and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['peakIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']

        return newItem

    # case 2 -> item1 peak and item2 peak are too close
    if abs(item1['peakIndex'] - item2['peakIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 3 -> item1 valley and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2
        # skip item2
        return newItem

    # case 4-> item1 valley is of similar height to item2 peak
    if abs(distance[item1['valleyIndex']] - distance[item2['peakIndex']]) < abs(
            distance[item1['valleyIndex']] - distance[item1['maxSpeedIndex']]) / 5:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item1['peakIndex']
            newItem['valleyIndex'] = item2['valleyIndex']

        return newItem

    return None


def compareNeighboursPositive(item1, item2, distance, minDistance=5):
    # case 1 -> item1 peak and item2 valley are too close
    if abs(item1['peakIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']

        return newItem

    # case 2 -> item1 peak and item2 peak are too close
    if abs(item1['peakIndex'] - item2['peakIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 3 -> item1 valley and item2 valley are too close
    if abs(item1['valleyIndex'] - item2['valleyIndex']) < minDistance:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = item1
        else:
            newItem = item2

        return newItem

    # case 4-> item1 valley is of similar height to item2 peak
    if abs(distance[item1['peakIndex']] - distance[item2['valleyIndex']]) < abs(
            distance[item1['peakIndex']] - distance[item1['maxSpeedIndex']]) / 5:
        # remove one of them, keep the one with highest speed
        if item1['maxSpeed'] > item2['maxSpeed']:
            newItem = {}
            newItem['maxSpeedIndex'] = item1['maxSpeedIndex']
            newItem['maxSpeed'] = item1['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']
        else:
            newItem = {}
            newItem['maxSpeedIndex'] = item2['maxSpeedIndex']
            newItem['maxSpeed'] = item2['maxSpeed']
            newItem['peakIndex'] = item2['peakIndex']
            newItem['valleyIndex'] = item1['valleyIndex']

        return newItem

    return None


def eliminateBadNeighboursNegative(indexVelocity, distance, minDistance=5):
    indexVelocityCorrected = []
    isSkip = [False] * len(indexVelocity)

    for idx in range(len(indexVelocity)):

        if isSkip[idx] == False:  # do not skip this item

            if idx < len(indexVelocity) - 1:

                newItem = compareNeighboursNegative(indexVelocity[idx], indexVelocity[idx + 1], distance, minDistance)
                if newItem is not None:
                    # newItem was returned, save returned element and skip following element
                    indexVelocityCorrected.append(newItem)
                    isSkip[idx + 1] = True
                else:
                    # no new Item, keep current item
                    indexVelocityCorrected.append(indexVelocity[idx])
            else:
                indexVelocityCorrected.append(indexVelocity[idx])

    return indexVelocityCorrected


def eliminateBadNeighboursPositive(indexVelocity, distance, minDistance=5):
    indexVelocityCorrected = []
    isSkip = [False] * len(indexVelocity)

    for idx in range(len(indexVelocity)):

        if isSkip[idx] == False:  # do not skip this item

            if idx < len(indexVelocity) - 1:

                newItem = compareNeighboursPositive(indexVelocity[idx], indexVelocity[idx + 1], distance,
                                                    minDistance=minDistance)
                if newItem is not None:
                    # newItem was returned, save returned element and skip following element
                    indexVelocityCorrected.append(newItem)
                    isSkip[idx + 1] = True
                else:
                    # no new Item, keep current item
                    indexVelocityCorrected.append(indexVelocity[idx])
            else:
                indexVelocityCorrected.append(indexVelocity[idx])

    return indexVelocityCorrected


def correctBasedonHeight(pos, distance, prct=0.125, minDistance=5):
    # eliminate any peaks that is smaller than 15% of the average height
    heightPeaks = []
    for item in pos:
        try:
            heightPeaks.append(abs(distance[item['peakIndex']] - distance[item['valleyIndex']]))
        except:
            pass

    meanHeightPeak = np.mean(heightPeaks)
    corrected = []
    for item in pos:
        try:
            if (abs(distance[item['peakIndex']] - distance[item['valleyIndex']])) > prct * meanHeightPeak:
                if abs(item['peakIndex'] - item['valleyIndex']) >= minDistance:
                    if (distance[item['peakIndex']] > distance[item['maxSpeedIndex']]) and (
                            distance[item['valleyIndex']] < distance[item['maxSpeedIndex']]):
                        corrected.append(item)
                    else:
                        pass
                else:
                    pass
            else:
                pass
        except:
            pass

    return corrected


def correctBasedonVelocityNegative(pos, velocity, prct=0.125):
    # velocity[velocity>0] = 0
    velocity = velocity ** 2

    velocityPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
        except:
            pass

    meanvelocityPeaks = np.mean(velocityPeaks)

    corrected = []
    for item in pos:
        try:
            if (velocity[item['maxSpeedIndex']]) > prct * meanvelocityPeaks:
                corrected.append(item)
            else:
                pass
        except:
            pass

    return corrected


def correctBasedonVelocityPositive(pos, velocity, prct=0.125):
    velocity[velocity < 0] = 0
    velocity = velocity ** 2

    velocityPeaks = []
    for item in pos:
        try:
            velocityPeaks.append(velocity[item['maxSpeedIndex']])
        except:
            pass

    meanvelocityPeaks = np.mean(velocityPeaks)

    corrected = []
    for item in pos:
        try:
            if (velocity[item['maxSpeedIndex']]) > prct * meanvelocityPeaks:
                corrected.append(item)
            else:
                pass
        except:
            pass

    return corrected


def correctFullPeaks(distance, pos, neg):
    # get the negatives
    closingVelocities = []
    for item in neg:
        closingVelocities.append(item['maxSpeedIndex'])

    openingVelocities = []
    for item in pos:
        openingVelocities.append(item['maxSpeedIndex'])

    peakCandidates = []
    for idx, closingVelocity in enumerate(closingVelocities):
        try:
            difference = np.array(openingVelocities) - closingVelocity
            difference[difference > 0] = 0

            posmin = np.argmax(difference[np.nonzero(difference)])

            absolutePeak = np.max(distance[pos[posmin]['maxSpeedIndex']: neg[idx]['maxSpeedIndex'] + 1])
            absolutePeakIndex = pos[posmin]['maxSpeedIndex'] + np.argmax(
                distance[pos[posmin]['maxSpeedIndex']: neg[idx]['maxSpeedIndex'] + 1])
            peakCandidate = {}

            peakCandidate['openingValleyIndex'] = pos[posmin]['valleyIndex']
            peakCandidate['openingPeakIndex'] = pos[posmin]['peakIndex']
            peakCandidate['openingMaxSpeedIndex'] = pos[posmin]['maxSpeedIndex']

            peakCandidate['closingValleyIndex'] = neg[idx]['valleyIndex']
            peakCandidate['closingPeakIndex'] = neg[idx]['peakIndex']
            peakCandidate['closingMaxSpeedIndex'] = neg[idx]['maxSpeedIndex']

            peakCandidate['peakIndex'] = absolutePeakIndex

            peakCandidates.append(peakCandidate)
        except:
            pass

    peakCandidatesCorrected = []
    idx = 0
    while idx < len(peakCandidates):

        peakCandidate = peakCandidates[idx]
        peak = peakCandidate['peakIndex']
        difference = [(peak - item['peakIndex']) for item in peakCandidates]
        if len(np.where(np.array(difference) == 0)[0]) == 1:
            peakCandidatesCorrected.append(peakCandidate)
            idx += 1
        else:
            item1 = peakCandidates[np.where(np.array(difference) == 0)[0][0]]
            item2 = peakCandidates[np.where(np.array(difference) == 0)[0][1]]
            peakCandidate = {}
            peakCandidate['openingValleyIndex'] = item1['openingValleyIndex']
            peakCandidate['openingPeakIndex'] = item1['openingPeakIndex']
            peakCandidate['openingMaxSpeedIndex'] = item1['openingMaxSpeedIndex']

            peakCandidate['closingValleyIndex'] = item2['closingValleyIndex']
            peakCandidate['closingPeakIndex'] = item2['closingPeakIndex']
            peakCandidate['closingMaxSpeedIndex'] = item2['closingMaxSpeedIndex']

            peakCandidate['peakIndex'] = item2['peakIndex']

            peakCandidatesCorrected.append(peakCandidate)
            idx += 2

    return peakCandidatesCorrected


def correctBasedonPeakSymmetry(peaks):
    peaksCorrected = []
    for peak in peaks:
        leftValley = peak['openingValleyIndex']
        centerPeak = peak['peakIndex']
        rightValley = peak['closingValleyIndex']

        ratio = (centerPeak - leftValley) / (rightValley - centerPeak)
        if 0.25 <= ratio <= 4:
            peaksCorrected.append(peak)

    return peaksCorrected


def peakFinder(rawSignal, fs=30, minDistance=5, cutOffFrequency=5, prct=0.125):
    indexPositiveVelocity = []
    indexNegativeVelocity = []

    b, a = signal.butter(2, cutOffFrequency, fs=fs, btype='lowpass', analog=False)

    distance = signal.filtfilt(b, a, rawSignal)  # signal.savgol_filter(rawDistance[0], 5, 3, deriv=0)
    velocity = signal.savgol_filter(distance, 5, 3, deriv=1) / (1 / fs)
    ##approx mean frequency
    acorr = np.convolve(rawSignal, rawSignal)
    t0 = ((1 / fs) * np.argmax(acorr))
    sep = 0.5 * (t0) if (0.5 * t0 > 1) else 1

    deriv = velocity.copy()
    deriv[deriv < 0] = 0
    deriv = deriv ** 2

    peaks, props = signal.find_peaks(deriv, distance=sep)

    heightPeaksPositive = deriv[peaks]
    selectedPeaksPositive = peaks[heightPeaksPositive > prct * np.mean(heightPeaksPositive)]

    # for each max opening vel, identify the peaks and valleys
    for idx, peak in enumerate(selectedPeaksPositive):
        idxValley = peak - 1
        if idxValley >= 0:
            while deriv[idxValley] != 0:
                if idxValley <= 0:
                    idxValley = np.nan
                    break

                idxValley -= 1

        idxPeak = peak + 1
        if idxPeak < len(deriv):
            while deriv[idxPeak] != 0:
                if idxPeak >= len(deriv) - 1:
                    idxPeak = np.nan
                    break

                idxPeak += 1

        if (not (np.isnan(idxPeak)) and not (np.isnan(idxValley))):
            positiveVelocity = {}
            positiveVelocity['maxSpeedIndex'] = peak
            positiveVelocity['maxSpeed'] = np.sqrt(deriv[peak])
            positiveVelocity['peakIndex'] = idxPeak
            positiveVelocity['valleyIndex'] = idxValley
            indexPositiveVelocity.append(positiveVelocity)

    deriv = velocity.copy()
    deriv[deriv > 0] = 0
    deriv = deriv ** 2
    peaks, props = signal.find_peaks(deriv, distance=sep)

    heightPeaksNegative = deriv[peaks]
    selectedPeaksNegative = peaks[heightPeaksNegative > prct * np.mean(heightPeaksNegative)]

    # for each max opening vel, identify the peaks and valleys
    for idx, peak in enumerate(selectedPeaksNegative):

        idxPeak = peak - 1
        if idxPeak >= 0:
            while deriv[idxPeak] != 0:
                if idxPeak <= 0:
                    idxPeak = np.nan
                    break

                idxPeak -= 1

        idxValley = peak + 1
        if idxValley < len(deriv):
            while deriv[idxValley] != 0:
                if idxValley >= len(deriv) - 1:
                    idxValley = np.nan
                    break

                idxValley += 1

        if (not (np.isnan(idxPeak)) and not (np.isnan(idxValley))):
            negativeVelocity = {}
            negativeVelocity['maxSpeedIndex'] = peak
            negativeVelocity['maxSpeed'] = np.sqrt(deriv[peak])
            negativeVelocity['peakIndex'] = idxPeak
            negativeVelocity['valleyIndex'] = idxValley
            indexNegativeVelocity.append(negativeVelocity)

            # euristics to remove bad peaks
    # # first, remove peaks that are too close to each other
    # indexPositiveVelocityCorrected = correctPeaksPositive(indexPositiveVelocity)    
    # indexNegativeVelocityCorrected = correctPeaksNegative(indexNegativeVelocity)
    # #then, remove peaks that are too small
    # indexPositiveVelocityCorrected = correctBasedonHeight(indexPositiveVelocityCorrected, distance)
    # indexNegativeVelocityCorrected = correctBasedonHeight(indexNegativeVelocityCorrected, distance)

    # remove bad peaks
    # 1- eliminate bad neighbours
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    # do it a couple of times
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    indexPositiveVelocity = eliminateBadNeighboursPositive(indexPositiveVelocity, distance, minDistance=minDistance)
    # 2-eliminate bad peaks based on height
    indexPositiveVelocity = correctBasedonHeight(indexPositiveVelocity, distance)
    # 3-eliminate bad peaks based on velocity
    indexPositiveVelocity = correctBasedonVelocityPositive(indexPositiveVelocity, velocity.copy())

    # 1- eliminate bad neighbours
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    # do it a couple of times
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    indexNegativeVelocity = eliminateBadNeighboursNegative(indexNegativeVelocity, distance, minDistance=minDistance)
    # 2-eliminate bad peaks based on height
    indexNegativeVelocity = correctBasedonHeight(indexNegativeVelocity, distance)
    # 3-eliminate bad peaks based on velocity
    indexNegativeVelocity = correctBasedonVelocityNegative(indexNegativeVelocity, velocity.copy())

    peaks = correctFullPeaks(distance, indexPositiveVelocity, indexNegativeVelocity)
    peaks = correctBasedonPeakSymmetry(peaks)

    return distance, velocity, peaks, indexPositiveVelocity, indexNegativeVelocity
