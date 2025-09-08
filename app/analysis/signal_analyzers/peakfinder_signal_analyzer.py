# analysis/core/signal_processor.py

import numpy as np
import scipy.signal as signal
import scipy.signal as signal
import scipy.interpolate as interpolate
from app.analysis.signal_analyzers.base_signal_analyzer import BaseSignalAnalyzer

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

        # 3) Peak finding
        distance, velocity, peaks, pos_vel, neg_vel = peakFinder(
            up_sample_signal,
            fs=up_fps,
            minDistance=3,
            cutOffFrequency=7.5,
            prct=0.05
        )

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
            p   = pk['peakIndex']
            vs  = pk['openingValleyIndex']
            ve  = pk['closingValleyIndex']

            # peak
            line_peaks.append(distance[p])
            line_peaks_time.append((p/size)*duration + start_time)
            # “middle” valley (same as opening)
            line_valleys.append(distance[vs])
            line_valleys_time.append((vs/size)*duration + start_time)
            # opening valley
            line_valleys_start.append(distance[vs])
            line_valleys_start_time.append((vs/size)*duration + start_time)
            # closing valley
            line_valleys_end.append(distance[ve])
            line_valleys_end_time.append((ve/size)*duration + start_time)

        # 6) Compute amplitudes, speeds, RMS, durations
        amplitude = []
        speed = []
        rms_vel = []
        avg_open_spd = []
        avg_close_spd = []
        cycle_dur = []
        peak_times = []

        for pk in peaks:
            vs = pk['openingValleyIndex']
            ve = pk['closingValleyIndex']
            p  = pk['peakIndex']

            y_vs = distance[vs]
            y_ve = distance[ve]
            y_p  = distance[p]

            # baseline at peak
            f_base = interpolate.interp1d([vs, ve], [y_vs, y_ve], fill_value="extrapolate")
            base_at_p = f_base(p)

            amp = y_p - base_at_p
            amplitude.append(amp)

            # RMS velocity over cycle
            vel_seg = velocity[vs:ve] if ve>vs else np.array([])
            rms = np.sqrt(np.mean(vel_seg**2)) if vel_seg.size else 0.0
            rms_vel.append(rms)

            # cycle duration in seconds
            cd = (ve - vs) / up_fps
            cycle_dur.append(cd if cd>0 else 1e-6)

            # overall speed
            speed.append(amp / cd if cd>0 else 0.0)

            # opening speed
            to = (p - vs) / up_fps
            avg_open_spd.append(amp / to if to>0 else 0.0)

            # closing speed
            tc = (ve - p) / up_fps
            avg_close_spd.append(amp / tc if tc>0 else 0.0)

            # record peak time (relative seconds)
            peak_times.append(p / up_fps)

        # 7) Statistics for radarTable
        def mean_std(arr):
            return (float(np.mean(arr)), float(np.std(arr))) if arr else (0.0, 0.0)

        meanAmp,  stdAmp  = mean_std(amplitude)
        meanSpd,  stdSpd  = mean_std(speed)
        meanRMS,  stdRMS  = mean_std(rms_vel)
        meanO,    stdO    = mean_std(avg_open_spd)
        meanC,    stdC    = mean_std(avg_close_spd)
        meanCd,   stdCd   = mean_std(cycle_dur)

        # range of cycle durations (max diff minus min diff)
        diffs = np.diff(peak_times)
        rangeCd = float(np.max(diffs) - np.min(diffs)) if diffs.size else 0.0

        # rate = cycles / (time from first opening to last closing)
        if peaks:
            span_frames = peaks[-1]['closingValleyIndex'] - peaks[0]['openingValleyIndex']
            span_s = span_frames / up_fps if span_frames>0 else 1e-6
            rate = len(peaks) / span_s
        else:
            rate = 0.0

        # decay metrics (early vs. late third)
        n3 = len(amplitude) // 3
        if n3 > 0:
            amp_dec = np.mean(amplitude[:n3]) / np.mean(amplitude[-n3:]) if np.mean(amplitude[-n3:]) else 1.0
            vel_dec = np.mean(speed[:n3])     / np.mean(speed[-n3:])     if np.mean(speed[-n3:])     else 1.0

            early = peaks[:n3]
            late  = peaks[-n3:]
            t_early = (early[-1]['closingValleyIndex'] - early[0]['openingValleyIndex'])/up_fps
            t_late  = (late[-1]['closingValleyIndex']  - late[0]['openingValleyIndex']) /up_fps
            rate_dec = ((len(early)/t_early)/(len(late)/t_late)) if t_early>0 and t_late>0 else 1.0
        else:
            amp_dec = vel_dec = rate_dec = 1.0

        # coefficient of variation
        cvAmp = stdAmp / meanAmp    if meanAmp else 0.0
        cvSpd = stdSpd / meanSpd    if meanSpd else 0.0
        cvCd  = stdCd  / meanCd     if meanCd  else 0.0
        cvR   = stdRMS / meanRMS    if meanRMS else 0.0
        cvO   = stdO   / meanO      if meanO   else 0.0
        cvC   = stdC   / meanC      if meanC   else 0.0

        # 8) Build final dict
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
            "radarTable": {
                "MeanAmplitude":           meanAmp,
                "StdAmplitude":            stdAmp,
                "MeanSpeed":               meanSpd,
                "StdSpeed":                stdSpd,
                "MeanRMSVelocity":         meanRMS,
                "StdRMSVelocity":          stdRMS,
                "MeanOpeningSpeed":        meanO,
                "stdOpeningSpeed":         stdO,
                "meanClosingSpeed":        meanC,
                "stdClosingSpeed":         stdC,
                "meanCycleDuration":       meanCd,
                "stdCycleDuration":        stdCd,
                "rangeCycleDuration":      rangeCd,
                "rate":                    rate,
                "amplitudeDecay":          amp_dec,
                "velocityDecay":           vel_dec,
                "rateDecay":               rate_dec,
                "cvAmplitude":             cvAmp,
                "cvCycleDuration":         cvCd,
                "cvSpeed":                 cvSpd,
                "cvRMSVelocity":           cvR,
                "cvOpeningSpeed":          cvO,
                "cvClosingSpeed":          cvC
            }
        }

        return jsonFinal



# ---------------------------------------------------------------
# -------------------- Peak Finder & Helpers --------------------
# ---------------------------------------------------------------
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
