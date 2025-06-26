import cv2
import numpy as np

def visualize_movement(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    features = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
    if features is None or len(features) < 5:
        return frame2.copy()
    new_features, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, features, None)
    good_old = features[status == 1]
    good_new = new_features[status == 1]
    vis_frame = frame2.copy()
    for i, (old, new) in enumerate(zip(good_old, good_new)):
        a, b = old.ravel()
        c, d = new.ravel()
        a, b, c, d = int(a), int(b), int(c), int(d)
        cv2.line(vis_frame, (a, b), (c, d), (0, 255, 0), 2)
        cv2.circle(vis_frame, (c, d), 5, (0, 0, 255), -1)
    return vis_frame

def visualize_object_movement(frame: np.ndarray, moving_pts: np.ndarray = None, hull: np.ndarray = None, min_area: int = 500) -> np.ndarray:
    vis_frame = frame.copy()
    
    if moving_pts is not None and len(moving_pts) > 0:
        for pt in moving_pts:
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(vis_frame, (x, y), 4, (0, 0, 255), -1)
   
    if hull is not None and len(hull) > 2:
        hull_int = hull.astype(np.int32)
        cv2.polylines(vis_frame, [hull_int], isClosed=True, color=(0, 255, 0), thickness=2)
       
        x, y, w, h = cv2.boundingRect(hull_int)
        if w * h > min_area:
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return vis_frame

def visualize_object_motion(frame: np.ndarray, motion_result: dict) -> np.ndarray:
    vis_frame = frame.copy()
    if motion_result['motion_detected']:
        cv2.drawContours(vis_frame, motion_result['contours'], -1, (0, 255, 0), 2)
        for cnt in motion_result['contours']:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        text = f"Objects: {motion_result['num_objects']}, Area: {motion_result['total_motion_area']:.0f}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    camera_motion = motion_result.get('camera_motion_magnitude', 0.0)
    compensated = motion_result.get('camera_motion_compensated', False)
    camera_text = f"Camera Motion: {camera_motion:.1f}"
    if compensated:
        camera_text += " (Compensated)"
    cv2.putText(vis_frame, camera_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    return vis_frame

def visualize_object_motion_bs(frame: np.ndarray, motion_result: dict) -> np.ndarray:
    vis_frame = frame.copy()
    if motion_result['motion_detected']:
       
        cv2.drawContours(vis_frame, motion_result['contours'], -1, (0, 255, 0), 2)
        
        for cnt in motion_result['contours']:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        text = f"Objects: {motion_result['num_objects']}, Area: {motion_result['total_motion_area']:.0f}"
        cv2.putText(vis_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return vis_frame 