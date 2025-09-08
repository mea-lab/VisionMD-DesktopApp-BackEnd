import numpy as np
import scipy.signal as signal
import scipy.signal as signal
import scipy.interpolate as interpolate
import tensorflow as tf
import tensorflow_hub as hub
from app.analysis.models.gait_transformer.gait_phase_transformer_old import load_default_model, get_gait_phase_stride_transformer, gait_phase_stride_inference
from app.analysis.models.gait_transformer.gait_phase_kalman import gait_kalman_smoother, compute_phases, get_event_times
import os
from django.conf import settings
from app.analysis.signal_analyzers.base_signal_analyzer import BaseSignalAnalyzer


class GaitSignalAnalyzer(BaseSignalAnalyzer):
    """
    Base signal analyzer for all signals (hand movement, finger tap, leg agility, toe tapping, etc.)
    Each  subclass must implement these abstract methods for analyzing a signal
    """

    # ------------------------------------------------------------------
    # --- START: Abstract properties definitions
    # ------------------------------------------------------------------

    _metrabs_joint_order = np.array(['htop', 'neck', 'rsho', 'relb', 'rwri', 'lsho',
                            'lelb', 'lwri', 'rhip', 'rkne', 'rank', 'lhip', 
                            'lkne', 'lank', 'pelv', 'spin', 'head'])
    
    _gait_phase_joint_order = ['pelv', 'rhip', 'rkne', 'rank', 'lhip', 'lkne', 
                            'lank', 'spin', 'neck', 'head', 'htop', 'lsho', 
                            'lelb', 'lwri', 'rsho', 'relb', 'rwri']
    
    _gait_phase_order_idx = None

    # ------------------------------------------------------------------
    # --- End: Abstract properties definitions
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # --- START: Abstract methods ---
    # ------------------------------------------------------------------
    def analyze(self, phases, strides, poses_3D, fps) -> dict:
        if GaitSignalAnalyzer._gait_phase_order_idx is None:
            GaitSignalAnalyzer._gait_phase_order_idx = np.array(
                [self._metrabs_joint_order.tolist().index(j) for j in GaitSignalAnalyzer._gait_phase_joint_order]
            )
    
        phase_ordered = np.take(phases, [0, 4, 1, 5, 2, 6, 3, 7], axis=-1)
        state, _, _ = gait_kalman_smoother(phase_ordered)
        timestamps = np.arange(state.shape[0])
        gait_event_dic = get_event_times(state, timestamps)

        results = self.analyze_gait_video_features(gait_event_dic, poses_3D, GaitSignalAnalyzer._gait_phase_order_idx, fps)

        return results, gait_event_dic


    # ------------------------------------------------------------------
    # --- END: Abstract methods ---
    # ------------------------------------------------------------------





    # ------------------------------------------------------------------
    # --- START: Helper methods ---
    # ------------------------------------------------------------------
    
    def analyze_gait_video_features(
        self,
        gait_event_dic: dict,
        keypoints_3D: np.ndarray,
        gait_phase_order_idx: list,
        fps,
    ) -> dict:
        """
        Analyze spatiotemporal gait features from gait event timings and 3D keypoints.

        Parameters:
            gait_event_dic (dict): Must contain four keys—
                'left_down', 'left_up', 'right_down', 'right_up'—
                each mapping to a 1D numpy array of event-frame floats.
            keypoints_3D (np.ndarray): Array of shape (n_frames, n_keypoints, 3)
                containing raw 3D keypoint coordinates.
            gait_phase_order_idx (list): Permutation index to reorder the keypoints.
            fps (int): Frames per second of the video.

        Returns:
            dict: A dictionary of lists (length 1) of gait features.
        """

        # --- 1) Extract raw event arrays ---
        lhs = np.asarray(gait_event_dic['left_down'],   dtype=float)
        ltf = np.asarray(gait_event_dic['left_up'],     dtype=float)
        rhs = np.asarray(gait_event_dic['right_down'],  dtype=float)
        rtf = np.asarray(gait_event_dic['right_up'],    dtype=float)
        if len(lhs) == 0 or len(ltf) == 0 or len(rhs) == 0 or len(rtf) == 0:
            print("Gait signal analyzing not passed")
            # raise ValueError("Error, no gait events were detected by gait transformer to compute features.")
            return {
            "Average stance time": 0,
            "Average swing time": 0,
            "Average double support time": 0,
            "Average step time": 0,
            "Average step length": 0,
            "Average velocity": 0,
            "Average cadence": 0,
            "Average stance time left": 0,
            "Average stance time right": 0,
            "Average swing time left": 0,
            "Average swing time right":  0,
            "Average step time left":      0,
            "Average step time right":   0,   
            "Average step length left":    0, 
            "Average step length right":    0,
            "Arm swing correlation":      0,
        }

        # --- 2) Temporal phases (in frames) ---
        
        # Calculate swing times
        if ltf[0] < lhs[0]:
            num_left_swings = min(len(ltf), len(lhs))
            L_swing  = lhs[0:num_left_swings] - ltf[0:num_left_swings]
        else:
            num_left_swings = min(len(ltf), len(lhs) - 1)
            L_swing  = lhs[1:num_left_swings + 1] - ltf[0:num_left_swings]
            
        if rtf[0] < rhs[0]:
            num_right_swings = min(len(rtf), len(rhs))
            R_swing  = rhs[0:num_right_swings] - rtf[0:num_right_swings]
        else:
            num_right_swings = min(len(rtf), len(rhs) - 1)
            R_swing  = rhs[1:num_right_swings + 1] - rtf[0:num_right_swings]
        if len(R_swing) == 0 or len(L_swing) == 0:
            raise ValueError("Error, no right swing or left swing events were detected by gait transformer to compute features.")

        # Calculate stance times
        if lhs[0] < ltf[0]:
            num_left_stances = min(len(lhs), len(ltf))
            L_stance  = ltf[0:num_left_stances] - lhs[0:num_left_stances]
        else:
            num_left_stances = min(len(lhs), len(ltf) - 1)
            L_stance  = ltf[1:num_left_stances + 1] - lhs[0:num_left_stances]

        if rhs[0] < rtf[0]:
            num_right_stances = min(len(rhs), len(rtf))
            R_stance  = rtf[0:num_right_stances] - rhs[0:num_right_stances]
        else:
            num_right_stances = min(len(rhs), len(rtf) - 1)
            R_stance  = rtf[1:num_right_stances + 1] - rhs[0:num_right_stances]
        if len(R_stance) == 0 or len(L_stance) == 0:
            raise ValueError("Error, no right stance or left stance events were detected by gait transformer to compute features.")

        # Calculate step times
        if rhs[0] < lhs[0]:
            num_left_steps = min(len(rhs), len(lhs))
            L_steptime = lhs[:num_left_steps] - rhs[:num_left_steps]
        else:
            num_left_steps = min(len(rhs), len(lhs) - 1)
            L_steptime = lhs[1:num_left_steps + 1] - rhs[:num_left_steps] 

        if lhs[0] < rhs[0]:
            num_right_steps = min(len(lhs), len(rhs))
            R_steptime = rhs[:num_right_steps] - lhs[:num_right_steps]
        else:
            num_right_steps = min(len(lhs), len(rhs) - 1)
            R_steptime = rhs[1:num_right_steps + 1] - lhs[:num_right_steps]
        if len(R_steptime) == 0 or len(L_steptime) == 0:
            raise ValueError("Error, not enough events were detected gait transformer to compute right steptime or left steptime features.")

        # Calculate double support times
        if lhs[0] < rtf[0]:
            num_d1 = min(len(lhs), len(rtf))
            d1 = rtf[:num_d1] - lhs[:num_d1]
        else:
            num_d1 = min(len(lhs), len(rtf) - 1)
            d1 = rtf[1:1 + num_d1] - lhs[:num_d1]

        if rhs[0] < ltf[0]:
            num_d2 = min(len(rhs), len(ltf))
            d2 = ltf[:num_d2] - rhs[:num_d2]
        else:
            num_d2 = min(len(rhs), len(ltf) - 1)
            d2 = ltf[1:1 + num_d2] - rhs[:num_d2]
        d1_trimmed = d1[:min(len(d1), len(d2))]
        d2_trimmed = d2[:min(len(d1), len(d2))]
        if len(d1_trimmed) == 0 or len(d2_trimmed) == 0:
            raise ValueError("Error, not enough events were detected gait transformer to compute the double support time feature.")

        # Combine for overall
        all_swings    = np.concatenate([L_swing, R_swing])
        all_stances   = np.concatenate([L_stance, R_stance])
        all_steptimes = np.concatenate([L_steptime, R_steptime])
        all_double_support = (d1_trimmed + d2_trimmed)[~np.isnan(d1_trimmed + d2_trimmed)]

        # --- 3) Convert to seconds & compute temporal averages ---
        avg_swing_left   = L_swing.mean()    / fps
        avg_swing_right  = R_swing.mean()    / fps
        avg_stance_left  = L_stance.mean()   / fps
        avg_stance_right = R_stance.mean()   / fps
        avg_steptime_left  = L_steptime.mean() / fps
        avg_steptime_right = R_steptime.mean() / fps

        avg_swing    = all_swings.mean()    / fps
        avg_stance   = all_stances.mean()   / fps
        avg_steptime = all_steptimes.mean() / fps
        avg_double   = all_double_support.mean() / fps
        cadence      = 60.0 / avg_steptime

        # --- 4) Spatial metrics from hip Z (reorder, scale to meters, flip Y) ---
        kp = keypoints_3D[:, gait_phase_order_idx] / 1000.0 
        kp[:, :, 1] *= -1.0
        z_hip = kp[:, 0, 2]

        lhs_idx = np.round(lhs).astype(int)
        rhs_idx = np.round(rhs).astype(int)

        # step lengths per side
        if lhs[0] < rhs[0]:
            m = min(len(lhs_idx) - 1, len(rhs_idx))
            sl_left  = np.abs(z_hip[lhs_idx[1:m+1]] - z_hip[rhs_idx[:m]])
            sl_right = np.abs(z_hip[rhs_idx[:m]]    - z_hip[lhs_idx[:m]])
        else:
            m = min(len(lhs_idx), len(rhs_idx) - 1)
            sl_left  = np.abs(z_hip[lhs_idx[:m]]      - z_hip[rhs_idx[:m]])
            sl_right = np.abs(z_hip[rhs_idx[1:m+1]]   - z_hip[lhs_idx[:m]])


        all_step_lengths = np.concatenate([sl_left, sl_right])
        strikes   = np.sort(np.concatenate([lhs_idx, rhs_idx]))
        avg_velocity = np.abs((z_hip[strikes[-1]] - z_hip[strikes[0]]) / (strikes[-1] - strikes[0]) * fps)

        # arm-swing correlation: elbows (keypoint idx 15 vs 12)
        kp_ctr   = kp - kp[:, 0:1]
        arm_corr = np.corrcoef(kp_ctr[:, 15, 2], kp_ctr[:, 12, 2])[0, 1]

        print("Gait signal analyzing passed")
        results = {
            "Average stance time":           float(avg_stance),
            "Average swing time":            float(avg_swing),
            "Average double support time":   float(avg_double),
            "Average step time":             float(avg_steptime),
            "Average step length":           float(all_step_lengths.mean()),
            "Average velocity":              float(avg_velocity),
            "Average cadence":               float(cadence),
            "Average stance time left":      float(avg_stance_left),
            "Average stance time right":     float(avg_stance_right),
            "Average swing time left":       float(avg_swing_left),
            "Average swing time right":      float(avg_swing_right),
            "Average step time left":        float(avg_steptime_left),
            "Average step time right":       float(avg_steptime_right),
            "Average step length left":      float(sl_left.mean()),
            "Average step length right":     float(sl_right.mean()),
            "Arm swing correlation":         float(arm_corr),
        }
        
        return results
    # ------------------------------------------------------------------
    # --- END: Helper methods ---
    # ------------------------------------------------------------------