import cv2
import numpy as np
import time
import os
from .video_detector import _detect_object_in_frame, _handle_no_features_frame, _check_quit, _display_progress_info

def _validate_input_files(query_img_path, video_path):
    """Validate that input files exist and are readable."""
    if not os.path.isfile(query_img_path):
        raise FileNotFoundError(f"Query image not found: {query_img_path}")
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")


def _preprocess_query_image(query_img_path):
    """Read and preprocess the query image."""
    query_img = cv2.imread(query_img_path)
    if query_img is None:
        raise ValueError(f"Failed to read query image: {query_img_path}")
    
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    # Apply histogram equalization to improve feature detection
    query_gray = cv2.equalizeHist(query_gray)
    
    return query_img, query_gray


def _create_output_path(video_path, save_output):
    """Create output filename if saving is enabled."""
    if save_output:
        filename, ext = os.path.splitext(video_path)
        return f"{filename}_detected{ext}"
    return None


def _initialize_video(video_path):
    """Initialize video capture and get video properties."""
    print(f"Using video file: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise IOError(f"Could not open video file: {video_path}")
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30  # Default to 30 fps if detection fails
        
    return cap, frame_width, frame_height, fps, total_frames


def _initialize_video_writer(save_output, output_path, fps, frame_width, frame_height):
    """Initialize video writer if saving is enabled."""
    if save_output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    return None


def _extract_query_features(feature_extractor, query_gray):
    """Extract features from the query image."""
    query_keypoints, query_descriptors = feature_extractor.detectAndCompute(query_gray, None)
    
    if query_keypoints is None or len(query_keypoints) == 0:
        raise ValueError("No keypoints found in query image. Try a different image.")
    
    print(f"Query image: {len(query_keypoints)} keypoints extracted")
    return query_keypoints, query_descriptors


def _initialize_matcher():
    """Initialize FLANN-based matcher for SIFT."""
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=8)
    search_params = dict(checks=100)  # Higher checks = more accurate but slower
    return cv2.FlannBasedMatcher(index_params, search_params)


def _setup_display_window(frame_width, frame_height, display_scale):
    """Create and setup display window."""
    window_name = "Object Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    window_width = int(frame_width * display_scale)
    window_height = int(frame_height * display_scale)
    cv2.resizeWindow(window_name, window_width, window_height)
    
    return window_name


def _process_video_frames(cap, feature_extractor, matcher, query_keypoints, query_descriptors,
                         query_gray, window_name, out, min_match_count, ratio_threshold,
                         ransac_threshold, frame_width, frame_height, total_frames):
    """Process each frame in the video for object detection."""
    # Track stats
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    detect_count = 0
    
    # Tracking variables
    last_valid_H = None
    object_detected_count = 0
    object_missed_count = 0
    
    print("Processing video. Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break
        
        # Update counters and stats
        frame_count += 1
        progress = frame_count / total_frames * 100 if total_frames > 0 else 0
        
        if frame_count % 30 == 0:
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames})")
            end_time = time.time()
            fps_display = 30 / (end_time - start_time)
            start_time = time.time()
        
        # Preprocess frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)
        result_frame = frame.copy()
        
        # Find keypoints and descriptors in current frame
        frame_keypoints, frame_descriptors = feature_extractor.detectAndCompute(frame_gray, None)
        
        # Skip if no features detected
        if frame_descriptors is None or len(frame_descriptors) == 0:
            _handle_no_features_frame(result_frame, window_name, out)
            if _check_quit():
                break
            continue
        
        # Detect object in current frame
        try:
            object_found, confidence_score, updated_H = _detect_object_in_frame(
                matcher, query_keypoints, query_descriptors, frame_keypoints, frame_descriptors,
                query_gray, result_frame, min_match_count, ratio_threshold, ransac_threshold,
                frame_width, frame_height, last_valid_H, object_missed_count
            )
            
            # Update tracking variables based on detection result
            if object_found:
                object_detected_count += 1
                object_missed_count = 0
                detect_count += 1
                last_valid_H = updated_H
            else:
                object_missed_count += 1
                
            # Reset tracking if we've missed too many frames
            if object_missed_count > 20:
                last_valid_H = None
                object_detected_count = 0
                
        except Exception as e:
            print(f"Error in frame {frame_count}: {str(e)}")
            cv2.putText(result_frame, f"Error: {str(e)[:30]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display progress information
        _display_progress_info(result_frame, frame_count, total_frames, fps_display, detect_count)
        
        # Display frame and check for quit
        cv2.imshow(window_name, result_frame)
        if out is not None:
            out.write(result_frame)
        
        if _check_quit():
            break
    
    return detect_count, frame_count


def _print_final_stats(frame_count, detect_count):
    """Print final statistics after processing."""
    detection_rate = (detect_count/frame_count*100) if frame_count > 0 else 0
    print(f"Processing complete. {frame_count} frames processed.")
    print(f"Object detected in {detect_count} frames ({detection_rate:.1f}%)")