import streamlit as st
import numpy as np
import cv2
import tempfile
import os
import pandas as pd
import io
import base64
import gc  # Garbage collection için
from PIL import Image
from core.movement import (
    detect_significant_movement,
    detect_significant_movement_orb,
    detect_object_movement,
    detect_object_movement_orb,
    detect_object_movement_with_compensation,
)
from core.visualization import visualize_movement, visualize_object_movement, visualize_object_motion_bs
from core.report import generate_report_link
from core.ui import sidebar_params, video_upload, main_tabs, show_results_camera, show_results_object, show_report_link

def analyze_orb_movement(frames, movement_indices, orb_threshold=5.0, min_matches=10):
    orb = cv2.ORB_create(nfeatures=500)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    prev_kp, prev_des = None, None
    prev_gray = None
    analysis = {}
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
                std_movement = np.std(magnitudes)
                if mean_movement > orb_threshold:
                    analysis[idx+1] = {
                        "Ortalama Hareket (px)": round(mean_movement,2),
                        "Standart Sapma": round(std_movement,2),
                        "Eşleşen Keypoint": len(matches)
                    }
        prev_kp, prev_des = kp, des
        prev_gray = gray
    return {idx: analysis.get(idx, {}) for idx in movement_indices}

def main():
    st.set_page_config(page_title="Camera and Object Movement Detection", layout="wide", initial_sidebar_state="expanded")
    st.title("Camera and Object Movement Detection")
    st.write("This application detects camera and object movements in videos.\n\n- **Camera movement** is detected using the ORB (Oriented FAST and Rotated BRIEF) feature matching algorithm.\n- **Object movement** is detected using background subtraction (MOG2) and contour area analysis.\n")
    tab1, tab2 = main_tabs()
    if 'results' not in st.session_state:
        st.session_state['results'] = None
    if 'analysis' not in st.session_state:
        st.session_state['analysis'] = None
    with tab1:
        detection_type, threshold, min_features, min_area, show_visualization = sidebar_params()
        uploaded_video = video_upload()
        start_analysis = st.button("Start Analysis")
        if uploaded_video and start_analysis:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            video_path = tfile.name
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            st.write(f"Video loaded: {total_frames} frames, {fps:.2f} FPS")
            # Deployment için optimize edilmiş frame limiti
            # Streamlit Cloud bellek limitlerini dikkate alarak
            is_cloud = (
                'STREAMLIT_SHARING' in os.environ or 
                'STREAMLIT_CLOUD' in os.environ or
                'HOSTNAME' in os.environ and 'streamlit' in os.environ.get('HOSTNAME', '').lower()
            )
            
            if is_cloud:
                # Cloud deployment için daha düşük limit
                MAX_FRAMES = 50  # Daha da düşük limit
                st.info("🚀 Cloud deployment tespit edildi. Performans için maksimum 50 frame analiz edilecek.")
            else:
                # Local çalıştırma için kullanıcı seçimi
                max_frames_option = st.selectbox(
                    "Maksimum frame sayısı seçin:",
                    ["50", "100", "200"],
                    index=1  # Varsayılan olarak 100
                )
                MAX_FRAMES = int(max_frames_option)
            
            if total_frames > MAX_FRAMES:
                sample_rate = max(1, total_frames // MAX_FRAMES)
                st.warning(f"Video çok uzun, bu nedenle sadece {MAX_FRAMES} frame analiz edilecek (her {sample_rate}. frame kullanılacak).")
            else:
                sample_rate = 1
            frames = []
            frame_indices = []
            with st.spinner(f"Extracting frames (sampling 1 in every {sample_rate} frames)..."):
                frame_idx = 0
                extracted_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if frame_idx % sample_rate == 0:
                        # Bellek optimizasyonu için frame'i küçült
                        if frame.shape[0] > 720:  # Yükseklik 720'den büyükse küçült
                            scale = 720 / frame.shape[0]
                            new_width = int(frame.shape[1] * scale)
                            frame = cv2.resize(frame, (new_width, 720))
                        frames.append(frame.copy())
                        frame_indices.append(frame_idx)
                        extracted_count += 1
                        
                        # Deployment için ekstra güvenlik - maksimum frame sayısını aşma
                        if extracted_count >= MAX_FRAMES:
                            break
                    frame_idx += 1
            cap.release()
            st.write(f"Extracted {len(frames)} frames for processing.")

            # Frame'leri ve analiz sonuçlarını session_state'e kaydet
            st.session_state['frames'] = frames
            st.session_state['frame_indices'] = frame_indices
            st.session_state['fps'] = fps
            st.session_state['uploaded_video'] = uploaded_video
            st.session_state['detection_type'] = detection_type
            st.session_state['threshold'] = threshold
            st.session_state['min_features'] = min_features
            st.session_state['min_area'] = min_area
            st.session_state['show_visualization'] = show_visualization

            key = (uploaded_video.name, detection_type, threshold, min_features, min_area)
            if 'last_key' not in st.session_state or st.session_state['last_key'] != key or start_analysis:
                if detection_type == "Camera Movement":
                    with st.spinner("Detecting camera movement..."):
                        movement_indices = detect_significant_movement_orb(
                            frames, threshold=threshold, min_matches=min_features
                        )
                        analysis = analyze_orb_movement(frames, movement_indices, orb_threshold=threshold, min_matches=min_features)
                    st.session_state['results'] = {
                        'type': 'camera',
                        'movement_indices': movement_indices,
                        'frame_indices': frame_indices,
                        'fps': fps,
                        'threshold': threshold,
                        'min_features': min_features,
                        'uploaded_video': uploaded_video,
                        'frames': frames,
                        'show_visualization': show_visualization
                    }
                    st.session_state['analysis'] = analysis
                    
                    # Bellek temizliği
                    gc.collect()
                else:
                    with st.spinner("Detecting object movement..."):
                        motion_results = detect_object_movement_with_compensation(
                            frames,
                            min_area=min_area
                        )
                    st.session_state['results'] = {
                        'type': 'object',
                        'motion_results': motion_results,
                        'frame_indices': frame_indices,
                        'fps': fps,
                        'min_area': min_area,
                        'uploaded_video': uploaded_video,
                        'frames': frames,
                        'show_visualization': show_visualization
                    }
                st.session_state['last_key'] = key
    with tab2:
        # Frame'leri ve analiz sonuçlarını session_state'ten al
        results = st.session_state.get('results', None)
        analysis = st.session_state.get('analysis', None)
        frames = st.session_state.get('frames', None)
        frame_indices = st.session_state.get('frame_indices', None)
        fps = st.session_state.get('fps', None)
        if results and frames is not None and frame_indices is not None and fps is not None:
            if results['type'] == 'camera':
                show_results_camera(
                    tab2,
                    results['movement_indices'],
                    frame_indices,
                    fps,
                    results['threshold'],
                    results['min_features'],
                    results['uploaded_video'],
                    generate_report_link,
                    frames,
                    results['show_visualization'],
                    visualize_movement
                )
                show_report_link(
                    tab2,
                    results['movement_indices'],
                    frame_indices,
                    fps,
                    results['threshold'],
                    results['min_features'],
                    None,
                    results['uploaded_video'],
                    generate_report_link,
                    'camera'
                )
            else:
                motion_results = results['motion_results']
                show_visualization = results['show_visualization']
                st.subheader("Frames with Detected Object Movement")
                detected_indices = [mr['frame_index'] for mr in motion_results if mr['motion_detected']]
                st.write(f"Object movement detected at {len(detected_indices)} frames.")
                st.write(f"Video frame indices: {detected_indices}")
                if detected_indices:
                    cols = st.columns(min(len(detected_indices), 3))
                    for i, idx in enumerate(detected_indices):
                        col_idx = i % len(cols)
                        with cols[col_idx]:
                            motion_result = motion_results[idx]
                            if show_visualization:
                                vis_frame = visualize_object_motion_bs(frames[idx], motion_result)
                                st.image(vis_frame, caption=f"Object Movement at frame {frame_indices[idx]}", use_container_width=True)
                            else:
                                st.image(frames[idx], caption=f"Object Movement at frame {frame_indices[idx]}", use_container_width=True)
                show_report_link(
                    tab2,
                    detected_indices,
                    frame_indices,
                    fps,
                    None,
                    None,
                    results['min_area'],
                    results['uploaded_video'],
                    generate_report_link,
                    'object'
                )
        else:
            st.info("Please upload a video and start analysis from the first tab.")

if __name__ == "__main__":
    main()
