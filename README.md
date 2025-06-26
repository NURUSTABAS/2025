# Camera and Object Movement Detection - Project Presentation

After uploading a video, you must click the **'Start Analysis'** button in the "Upload & Analysis" tab to perform the analysis.
If you change any parameter or switch between camera movement and object movement detection, you need to click the **'Start Analysis'** button again to re-analyze the video with the new settings.
This ensures that the analysis is only performed when you explicitly request it, preventing unnecessary computations for large videos.
After the analysis is completed, the captured frames (analyzed video frames) and detected movements are visually presented in the Results tab.
Additionally, at the bottom of the Results section, there is a downloadable analysis report available for the user.

+Note: On cloud deployments (such as Streamlit Cloud), the number of frames analyzed is limited for performance reasons. On local runs, you can select the maximum number of frames to analyze according to your system's capabilities.
## ☁️ Live Demo & User Video

 Open the Live Streamlit App -- [https://2025-core-talent.streamlit.app](https://core2025.streamlit.app/)  

## 1. Project Overview

This project is a Streamlit web application for detecting both **camera movement** and **object movement** in video files. Camera movement detection uses the ORB (Oriented FAST and Rotated BRIEF) feature matching algorithm, while object movement detection is based on background subtraction (MOG2) and contour area analysis. Users can upload videos, adjust detection parameters, visualize results, and download detailed CSV reports of detected movement events.

### 1.1. Main Features

- **Two Types of Movement Detection:** Camera movement and object movement
- **Video File Support:** Upload MP4, AVI, or MOV files
- **Adjustable Parameters:** User-friendly parameter controls
- **Visualization:** Visual overlays for detected movements
- **Reporting:** Downloadable CSV analysis reports

## 2. Movement Detection Logic

### 2.1. Camera Movement Detection

- **Technology:** ORB feature extraction and matching
- **Parameters:**
  - **Movement Threshold:** Minimum average movement (in pixels) between matched keypoints to detect camera movement
  - **Minimum Features:** Minimum number of matched ORB keypoints required for reliable detection
- **Logic:** For each pair of consecutive frames, ORB keypoints are extracted and matched. If the number of matches exceeds the minimum and the average movement exceeds the threshold, the frame is marked as a camera movement event.

### 2.2. Object Movement Detection

- **Technology:** Background subtraction (MOG2) and contour area analysis
- **Parameter:**
  - **Minimum Object Area:** Minimum area (in pixels) for a detected contour to be considered as object movement
- **Logic:** For each frame, a foreground mask is generated using MOG2. Contours are extracted, and if any contour's area exceeds the minimum, the frame is marked as containing object movement. Frames with significant camera movement are excluded from object movement results.

## 3. Parameters and Their Effects

- **Camera Movement:**
  - _Movement Threshold_: Controls sensitivity to camera movement (higher = less sensitive)
  - _Minimum Features_: Ensures reliability by requiring enough keypoint matches
- **Object Movement:**
  - _Minimum Object Area_: Filters out small, insignificant movements

Changing any parameter or switching between detection types does **not** automatically re-run the analysis. You must click the **'Start Analysis'** button in the "Upload & Analysis" tab to update the results with the new settings.

## 4. Core Code Snippets

### Camera Movement Detection (ORB)

```python
def detect_significant_movement_orb(frames, threshold=5.0, min_matches=10):
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_des = None, None
    movement_indices = []
    for idx, frame in enumerate(frames[:-1]):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        if prev_kp is not None and prev_des is not None and des is not None:
            matches = bf.match(prev_des, des)
            if len(matches) >= min_matches:
                pts_prev = np.float32([prev_kp[m.queryIdx].pt for m in matches])
                pts_curr = np.float32([kp[m.trainIdx].pt for m in matches])
                mean_movement = np.mean(np.linalg.norm(pts_curr - pts_prev, axis=1))
                if mean_movement > threshold:
                    movement_indices.append(idx + 1)
        prev_kp, prev_des = kp, des
    return movement_indices
```

### Object Movement Detection (Background Subtraction)

```python
def detect_object_movement_with_compensation(frames, min_area=500, camera_motion_threshold=5.0):
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
    motion_results = []
    prev_frame = None
    for idx, frame in enumerate(frames):

        fg_mask = bg_subtractor.apply(frame)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]

        motion_detected = len(valid_contours) > 0 # and not camera_motion_detected
        motion_results.append({'frame_index': idx, 'motion_detected': motion_detected})
    return motion_results

```

## 5. Video Processing Workflow

- User uploads a video file
- Frames are extracted and analyzed
- Camera and/or object movement is detected based on selected mode and parameters
- Results are visualized and a downloadable CSV report is generated

## 6. Visualization

- Camera movement: Visual overlays between frames with detected movement
- Object movement: Bounding boxes on moving objects (excluding frames with significant camera movement)

## 7. Reporting System

- Downloadable CSV report at the bottom of the Results page
- **Camera movement report columns:** Frame Index, Time (seconds), Movement Threshold Used, Min Features Used
- **Object movement report columns:** Frame Index, Time (seconds), Min Object Area Used

## 8. Use Cases

- Video production: Detecting camera shake, scene transitions
- Security: Detecting camera tampering or sabotage
- Robotics/Drone: Evaluating camera stabilization, navigation analysis

## 9. Technical Requirements

- **Python 3.6+**
- **Main Libraries:**
  - streamlit
  - opencv-python
  - numpy
  - pandas
  - Pillow

## 10. Project Structure

```
camera-movement-detection/
├── app.py                    # Main Streamlit application
├── core/
│   ├── movement.py           # Movement detection algorithms
│   ├── report.py             # CSV report generation
│   ├── ui.py                 # UI components and helpers
│   └── visualization.py      # Visualization utilities
├── DEPLOYMENT.md             # Deployment instructions
├── Dockerfile                # Docker configuration
├── PROJECT_PRESENTATION.md   # This presentation file
└── README.md                 # Project README
```

## 11. Assumptions and Challenges

**Assumptions:**

- Input videos are of reasonable quality and not extremely noisy.
- Object movement and camera movement do not always occur simultaneously.

**Challenges:**

- Distinguishing between camera movement and object movement, especially when both occur together.
- Avoiding false positives in object movement detection due to camera shake.
- Ensuring reliable detection with varying video quality and lighting conditions.
- Providing a responsive and user-friendly interface where the user explicitly triggers analysis by clicking the 'Start Analysis' button after changing parameters or detection type.

## 12. How to Run the App Locally

```bash
git clone https://github.com/NURUSTABAS/2025

cd .\2025-main\camera-movement-detection\

pip install -r requirements.txt
venv\Scripts\activate
streamlit run app.py
```

## 13. Input Output images
streamlit.app  Screen 
![image](https://github.com/user-attachments/assets/8784f324-6f43-41d2-9290-baa68fc3cabd)
![image](https://github.com/user-attachments/assets/ce903240-9ade-431a-9929-2869c7d3d3f6)



Local Screen 
![image](https://github.com/user-attachments/assets/e40519e4-c302-4966-bd2a-7145b0bbd13d)

![image](https://github.com/user-attachments/assets/a502cd43-0cc2-4bc1-b441-51d1004bcf95)

![image](https://github.com/user-attachments/assets/fe386a8f-3fcc-45c7-939f-8ae86927fa05)


## AI Prompts & Tools Used

- **ChatGPT**: Assisted with OpenCV algorithm explanations, code improvements, and documentation structure.
- **Cursor**: Used as an AI-powered code editor for efficient refactoring, debugging, and performance comparisons (e.g., SIFT vs. ORB), as well as resolving Streamlit deployment issues.

---
