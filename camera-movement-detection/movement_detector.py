import cv2
import numpy as np
from typing import List, Tuple, Optional

def detect_significant_movement(frames: List[np.ndarray], threshold: float = 0.5, min_features: int = 10) -> List[int]:
   
    movement_indices = []
    prev_gray = None
    prev_features = None
    
    for idx, frame in enumerate(frames[:-1]): 
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_gray is None:

            prev_gray = gray
            prev_features = cv2.goodFeaturesToTrack(
                prev_gray, 
                maxCorners=100,  
                qualityLevel=0.3, 
                minDistance=7, 
                blockSize=7
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
            gray, 
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
    
    return movement_indices

def calculate_movement_score(old_features: np.ndarray, new_features: np.ndarray) -> float:
   
    
    displacements = new_features - old_features
    
   
    magnitudes = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
    
    
    median_displacement = np.median(magnitudes)
    
   
    angles = np.arctan2(displacements[:, 1], displacements[:, 0])
    angle_variance = np.var(angles)
    
    movement_score = median_displacement * (1 - min(angle_variance / np.pi, 0.5))
    
    return movement_score

def visualize_movement(frame1: np.ndarray, frame2: np.ndarray) -> np.ndarray:
    """
    Create a visualization of movement between two frames.
    
    Args:
        frame1: First frame.
        frame2: Second frame.
        
    Returns:
        Frame with movement visualization.
    """
    
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

def detect_object_movement(frames: List[np.ndarray], min_area: int = 500) -> List[int]:
  
    movement_indices = []
    fgbg = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=16, detectShadows=True)
    for idx, frame in enumerate(frames):
        fgmask = fgbg.apply(frame)
       
        _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
       
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
      
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                movement_indices.append(idx)
                break
    return movement_indices

def visualize_object_movement(frame: np.ndarray, min_area: int = 500) -> np.ndarray:
    """
    Visualize moving objects in a frame using background subtraction.
    Args:
        frame: Input frame (numpy array).
        min_area: Minimum area for a contour to be considered a moving object.
    Returns:
        Frame with bounding boxes drawn around moving objects.
    """
    fgbg = cv2.createBackgroundSubtractorMOG2(history=1, varThreshold=16, detectShadows=True)
    fgmask = fgbg.apply(frame)
    _, fgmask = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    vis_frame = frame.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return vis_frame

def detect_significant_movement_orb(frames: List[np.ndarray], threshold: float = 5.0, min_matches: int = 10) -> List[int]:
    """
    ORB ve brute-force matcher ile ardışık kareler arasında hareket tespiti yapar.
    threshold: Ortalama hareket vektörü uzunluğu eşiği.
    min_matches: Güvenilirlik için minimum eşleşme sayısı.
    """
    movement_indices = []
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_des = None, None
    prev_gray = None
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
        prev_gray = gray
    return movement_indices

def detect_object_movement_orb(frames: List[np.ndarray], min_matches: int = 10, min_area: int = 500) -> List[int]:
    
    movement_indices = []
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_des = None, None
    prev_gray = None
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
                
                moving_points = magnitudes > 10  
                if np.sum(moving_points) > 0:
                   
                    moving_pts = pts_curr[moving_points]
                    if len(moving_pts) >= 3:
                        hull = cv2.convexHull(moving_pts)
                        area = cv2.contourArea(hull)
                        if area > min_area:
                            movement_indices.append(idx + 1)
        prev_kp, prev_des = kp, des
        prev_gray = gray
    return movement_indices
