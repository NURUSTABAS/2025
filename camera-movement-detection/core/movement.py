import cv2
import numpy as np
from typing import List, Tuple

def estimate_camera_motion(frame1: np.ndarray, frame2: np.ndarray) -> Tuple[np.ndarray, float]:
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    features1 = cv2.goodFeaturesToTrack(gray1, maxCorners=200, qualityLevel=0.01, minDistance=10)
    if features1 is None or len(features1) < 10:
        return np.eye(3), 0.0
    features2, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, features1, None)
    good_old = features1[status == 1]
    good_new = features2[status == 1]
    if len(good_old) < 10:
        return np.eye(3), 0.0
    H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, 5.0)
    if H is None:
        return np.eye(3), 0.0
    tx = H[0, 2]
    ty = H[1, 2]
    motion_magnitude = np.sqrt(tx**2 + ty**2)
    return H, motion_magnitude

def detect_significant_camera_movement(frames: List[np.ndarray], camera_motion_threshold: float = 5.0) -> List[int]:
    movement_indices = []
    prev_frame = None
    for idx, frame in enumerate(frames):
        if prev_frame is not None:
            _, motion_magnitude = estimate_camera_motion(prev_frame, frame)
            if motion_magnitude > camera_motion_threshold:
                movement_indices.append(idx)
        prev_frame = frame
    return movement_indices

def compensate_camera_motion(frame: np.ndarray, homography: np.ndarray) -> np.ndarray:
    if homography is None or np.array_equal(homography, np.eye(3)):
        return frame
    h, w = frame.shape[:2]
    return cv2.warpPerspective(frame, homography, (w, h))

def detect_object_movement_with_compensation(
    frames: List[np.ndarray],
    history: int = 500,
    var_threshold: float = 16.0,
    detect_shadows: bool = True,
    min_area: int = 500,
    max_area: int = 50000,
    camera_motion_threshold: float = 5.0
) -> List[dict]:
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=var_threshold,
        detectShadows=detect_shadows
    )
    motion_results = []
    prev_frame = None
    prev_homography = np.eye(3)
    for idx, frame in enumerate(frames):
        camera_motion_magnitude = 0.0
        if prev_frame is not None:
            homography, camera_motion_magnitude = estimate_camera_motion(prev_frame, frame)
            if camera_motion_magnitude > camera_motion_threshold:
                frame = compensate_camera_motion(frame, homography)
                prev_homography = homography
            else:
                frame = compensate_camera_motion(frame, prev_homography)
        else:
            frame = compensate_camera_motion(frame, np.eye(3))
        fg_mask = bg_subtractor.apply(frame)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
        if camera_motion_magnitude > camera_motion_threshold:
            motion_detected = False
        else:
            motion_detected = len(valid_contours) > 0
        result = {
            'frame_index': idx,
            'motion_detected': motion_detected,
            'num_objects': len(valid_contours),
            'total_motion_area': sum(cv2.contourArea(cnt) for cnt in valid_contours),
            'contours': valid_contours,
            'foreground_mask': fg_mask,
            'camera_motion_magnitude': camera_motion_magnitude,
            'camera_motion_compensated': camera_motion_magnitude > camera_motion_threshold
        }
        motion_results.append(result)
        prev_frame = frames[idx]
    return motion_results

def detect_significant_movement(frames: List[np.ndarray], threshold: float = 0.5, min_features: int = 10) -> List[int]:
    movement_indices = []
    prev_gray = None
    prev_features = None
    for idx, frame in enumerate(frames[:-1]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            prev_gray = gray
            prev_features = cv2.goodFeaturesToTrack(
                prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
            )
            continue
        if prev_features is not None and len(prev_features) >= min_features:
            new_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, prev_features, None
            )
            good_old = prev_features[status == 1]
            good_new = new_features[status == 1]
            if len(good_old) >= min_features and len(good_new) >= min_features:
                movement_score = calculate_movement_score(good_old, good_new)
                if movement_score > threshold:
                    movement_indices.append(idx + 1)
        prev_gray = gray
        prev_features = cv2.goodFeaturesToTrack(
            gray, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7
        )
    return movement_indices

def detect_significant_movement_orb(frames: List[np.ndarray], threshold: float = 5.0, min_matches: int = 10) -> List[int]:
    
    movement_indices = []
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_des = None, None
    for idx, frame in enumerate(frames[:-1]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if prev_kp is not None and prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            if len(matches) >= min_matches:
                pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])
                displacements = pts_curr - pts_prev
                magnitudes = np.linalg.norm(displacements, axis=1)
                mean_movement = np.mean(magnitudes)
                if mean_movement > threshold:
                    movement_indices.append(idx + 1)
        prev_kp, prev_des = kp, des
    return movement_indices

def detect_object_movement(frames: List[np.ndarray], min_area: int = 500, history: int = 50, varThreshold: int = 16, detectShadows: bool = True) -> List[int]:
    movement_indices = []
    fgbg = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold, detectShadows=detectShadows)
    for idx, frame in enumerate(frames):
        fgmask = fgbg.apply(frame)
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                movement_indices.append(idx)
                break
    return movement_indices

def detect_object_movement_orb(frames: List[np.ndarray], min_matches: int = 10, min_area: int = 500, movement_threshold: float = 10.0) -> List[int]:
   
    movement_indices = []
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_des = None, None
    for idx, frame in enumerate(frames[:-1]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if prev_kp is not None and prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            if len(matches) >= min_matches:
                pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])
                displacements = pts_curr - pts_prev
                magnitudes = np.linalg.norm(displacements, axis=1)
                moving_points = magnitudes > movement_threshold
                if np.sum(moving_points) > 0:
                    moving_pts = pts_curr[moving_points]
                    if len(moving_pts) >= 3:
                        hull = cv2.convexHull(moving_pts)
                        area = cv2.contourArea(hull)
                        if area > min_area:
                            movement_indices.append(idx + 1)
        prev_kp, prev_des = kp, des
    return movement_indices

def calculate_movement_score(old_features: np.ndarray, new_features: np.ndarray) -> float:
    displacements = new_features - old_features
    magnitudes = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
    median_displacement = np.median(magnitudes)
    angles = np.arctan2(displacements[:, 1], displacements[:, 0])
    angle_variance = np.var(angles)
    movement_score = median_displacement * (1 - min(angle_variance / np.pi, 0.5))
    return movement_score 