import streamlit as st

def sidebar_params():
    st.sidebar.header("Detection Type")
    detection_type = st.sidebar.radio(
        "Which type of movement do you want to detect?",
        ["Camera Movement", "Object Movement"]
    )
    threshold = 5.0
    min_features = 10
    min_area = 500
    show_visualization = True

    if detection_type == "Camera Movement":
        threshold = st.sidebar.slider(
            "Movement Threshold",
            min_value=2.0,
            max_value=10.0,
            value=5.0,
            step=0.1,
            help="Increase for detecting only large camera movements."
        )
        min_features = st.sidebar.slider(
            "Minimum Features (Camera)",
            min_value=5,
            max_value=50,
            value=10,
            step=1,
            help="Minimum number of ORB matches required between frames for camera movement detection."
        )
        min_area = 500 
    elif detection_type == "Object Movement":
        min_area = st.sidebar.slider(
            "Minimum Object Area (pixels)",
            min_value=200, max_value=2000, value=500, step=10,
            help="Minimum area of detected objects. Larger values filter out small movements."
        )
        min_features = 10
        threshold = 10.0

    show_visualization = st.sidebar.checkbox(
        "Show Movement Visualization",
        value=True,
        help="Show visual overlay for detected movement in frames."
    )
    return detection_type, threshold, min_features, min_area, show_visualization

def video_upload():
    uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    return uploaded_video

def main_tabs():
    tab1, tab2 = st.tabs(["📤 Upload & Analysis", "📊 Results"])
    return tab1, tab2

def show_results_camera(tab, movement_indices, frame_indices, fps, threshold, min_features, uploaded_video, generate_report_link, frames, show_visualization, visualize_movement):
    video_movement_indices = [frame_indices[i] for i in movement_indices]
    with tab:
        st.subheader("Frames with Detected Camera Movement and Analysis")
        st.write(f"Significant camera movement detected at {len(movement_indices)} frames.")
        st.write(f"Video frame indices: {video_movement_indices}")
        if movement_indices:
            cols = st.columns(min(len(movement_indices), 3))
            for i, idx in enumerate(movement_indices):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if idx > 0 and show_visualization and idx < len(frames):
                        vis_frame = visualize_movement(frames[idx-1], frames[idx])
                        st.image(vis_frame, caption=f"Camera Movement at frame {frame_indices[idx]}", use_container_width=True)
                    elif idx < len(frames):
                        st.image(frames[idx], caption=f"Camera Movement at frame {frame_indices[idx]}", use_container_width=True)

def show_results_object(tab, movement_indices, frame_indices, fps, min_area, uploaded_video, generate_report_link, frames, show_visualization, visualize_object_movement):
    video_movement_indices = [frame_indices[i] for i in movement_indices]
    with tab:
        st.subheader("Frames with Detected Object Movement")
        st.write(f"Object movement detected at {len(movement_indices)} frames.")
        st.write(f"Video frame indices: {video_movement_indices}")
        if movement_indices:
            cols = st.columns(min(len(movement_indices), 3))
            for i, idx in enumerate(movement_indices):
                col_idx = i % len(cols)
                with cols[col_idx]:
                    if show_visualization:
                        vis_frame = visualize_object_movement(frames[idx], min_area=min_area)
                        st.image(vis_frame, caption=f"Object Movement at frame {frame_indices[idx]}", use_container_width=True)
                    else:
                        st.image(frames[idx], caption=f"Object Movement at frame {frame_indices[idx]}", use_container_width=True)

def show_report_link(tab, movement_indices, frame_indices, fps, threshold, min_features, min_area, uploaded_video, generate_report_link, result_type):
    video_movement_indices = [frame_indices[i] for i in movement_indices]
    with tab:
        st.subheader("Analysis Report")
        if result_type == 'camera':
            if movement_indices:
                report_data = {
                    "Frame Index": video_movement_indices,
                    "Time (seconds)": [idx/fps for idx in video_movement_indices],
                    "Movement Threshold Used": [threshold] * len(movement_indices),
                    "Min Features Used": [min_features] * len(movement_indices)
                }
                st.markdown(generate_report_link(report_data, f"{uploaded_video.name}_camera_movement_analysis.csv"), unsafe_allow_html=True)
        else:
            if movement_indices:
                report_data = {
                    "Frame Index": video_movement_indices,
                    "Time (seconds)": [idx/fps for idx in video_movement_indices],
                    "Min Object Area Used": [min_area] * len(movement_indices)
                }
                st.markdown(generate_report_link(report_data, f"{uploaded_video.name}_object_movement_analysis.csv"), unsafe_allow_html=True) 