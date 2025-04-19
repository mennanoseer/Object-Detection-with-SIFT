import cv2
import argparse
from helpers.video_processor import (
    _validate_input_files, _preprocess_query_image, _create_output_path,
    _initialize_video, _initialize_video_writer, _extract_query_features,
    _initialize_matcher, _setup_display_window, _process_video_frames,
    _print_final_stats
)

def detect_object_in_video(query_img_path, video_path, min_match_count=10,
                           display_scale=1.0, ratio_threshold=0.7, ransac_threshold=5.0,
                           save_output=False):
    """
    Detect an object from a query image in a video file using SIFT feature matching.
    
    Parameters:
    - query_img_path: Path to the query image containing the object to find
    - video_path: Path to the video file
    - min_match_count: Minimum number of good matches required
    - display_scale: Scale factor for display
    - ratio_threshold: Threshold for Lowe's ratio test (lower = more strict)
    - ransac_threshold: RANSAC threshold for homography estimation
    - save_output: Whether to save the processed video
    """
    # Validate input files
    _validate_input_files(query_img_path, video_path)
    
    # Read and preprocess query image
    query_img, query_gray = _preprocess_query_image(query_img_path)
    
    # Create output filename if saving
    output_path = _create_output_path(video_path, save_output)
    
    # Initialize video capture and get properties
    cap, frame_width, frame_height, fps, total_frames = _initialize_video(video_path)
    
    # Initialize output video writer if requested
    out = _initialize_video_writer(save_output, output_path, fps, frame_width, frame_height)
    
    # Initialize SIFT feature extractor and extract query features
    feature_extractor = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03, edgeThreshold=10)
    query_keypoints, query_descriptors = _extract_query_features(feature_extractor, query_gray)
    
    # Initialize FLANN-based matcher for SIFT
    matcher = _initialize_matcher()
    
    # Create display window
    window_name = _setup_display_window(frame_width, frame_height, display_scale)
    
    # Process video frames
    detect_count, frame_count = _process_video_frames(
        cap, feature_extractor, matcher, query_keypoints, query_descriptors, 
        query_gray, window_name, out, min_match_count, ratio_threshold, 
        ransac_threshold, frame_width, frame_height, total_frames
    )
    
    # Release resources
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()
    
    if save_output:
        print(f"Output video saved to: {output_path}")
    
    # Print final statistics
    _print_final_stats(frame_count, detect_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect an object in a video using SIFT feature matching')
    parser.add_argument('--query', type=str, required=True, help='Path to query image')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    parser.add_argument('--min-matches', type=int, default=10, help='Minimum good matches required')
    parser.add_argument('--scale', type=float, default=1.0, help='Display scale factor')
    parser.add_argument('--ratio', type=float, default=0.7, help='Lowe\'s ratio test threshold')
    parser.add_argument('--ransac', type=float, default=5.0, help='RANSAC threshold for homography')
    parser.add_argument('--save', action='store_true', help='Save output video')
    
    args = parser.parse_args()
    
    try:
        detect_object_in_video(
            query_img_path=args.query, 
            video_path=args.video, 
            min_match_count=args.min_matches,
            display_scale=args.scale,
            ratio_threshold=args.ratio,
            ransac_threshold=args.ransac,
            save_output=args.save
        )
    except Exception as e:
        print(f"Error: {str(e)}")