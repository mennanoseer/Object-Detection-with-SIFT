import cv2
import numpy as np
import os
from utils import validate_homography
from image_processor import load_image, create_feature_extractor, create_feature_matcher, filter_matches

# Define color palette for different objects (BGR format)
COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (0, 255, 255),  # Yellow
    (255, 0, 255),  # Magenta
    (255, 255, 0),  # Cyan
    (128, 0, 0),    # Dark blue
    (0, 128, 0),    # Dark green
    (0, 0, 128),    # Dark red
    (128, 128, 0),  # Teal
]

def detect_multiple_objects(query_img_paths, target_img_path, min_match_count=10, output_path=None, display=True):
    """
    Detect multiple objects from query images in a target image using SIFT feature matching.
    
    Parameters:
    - query_img_paths: List of paths to query images containing objects to find
    - target_img_path: Path to the target image where we need to find the objects
    - min_match_count: Minimum number of good matches required
    - output_path: Path to save the result image (if None, image is not saved)
    - display: Whether to display the results
    
    Returns:
    - Result image with all objects highlighted
    - Dictionary mapping query image paths to detection status (True/False)
    """
    from utils import display_multiple_results
    
    # Check if target file exists
    if not os.path.isfile(target_img_path):
        raise FileNotFoundError(f"Target image not found: {target_img_path}")
        
    # Read the target image
    target_img = load_image(target_img_path)
    
    if target_img is None:
        raise ValueError(f"Could not read target image: {target_img_path}")
    
    # Convert target to grayscale
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    
    # Create a copy of the target image for highlighting all objects
    result_img = target_img.copy()
    
    # Initialize SIFT feature extractor
    feature_extractor = create_feature_extractor()
    
    # Find keypoints and descriptors for target image (do this once)
    target_keypoints, target_descriptors = feature_extractor.detectAndCompute(target_gray, None)
    
    # Dictionary to track detection results
    detection_results = {}
    
    # Process each query image
    for idx, query_img_path in enumerate(query_img_paths):
        # Get color for this object
        color = COLORS[idx % len(COLORS)]
        
        # Detect the object
        object_found = detect_single_object(
            query_img_path, 
            target_img, 
            target_gray, 
            target_keypoints, 
            target_descriptors, 
            result_img,
            feature_extractor,
            color,
            min_match_count
        )
        
        detection_results[query_img_path] = object_found
    
    # Save the result if output path is provided
    if output_path and any(detection_results.values()):
        print(f"Saving result to {output_path}")
        cv2.imwrite(output_path, result_img)
    
    # Display the results if requested
    if display:
        display_multiple_results(query_img_paths, target_img_path, result_img)
    
    return result_img, detection_results

def detect_single_object(query_img_path, target_img, target_gray, target_keypoints, 
                        target_descriptors, result_img, feature_extractor, color, min_match_count):
    """
    Detect a single object from query image in a target image.
    
    Parameters:
    - query_img_path: Path to the query image containing object to find
    - target_img: Target image as numpy array
    - target_gray: Grayscale version of target image
    - target_keypoints: Keypoints from target image
    - target_descriptors: Descriptors from target image
    - result_img: Image to draw results on
    - feature_extractor: SIFT or other feature extractor
    - color: Color to use for this object (BGR format)
    - min_match_count: Minimum number of good matches required
    
    Returns:
    - Boolean indicating if object was found
    """
    # Get a lighter version of the color for highlighting
    highlight_color = tuple(min(c + 50, 255) for c in color)
    
    # Check if query file exists and load it
    query_img = load_image(query_img_path)
    if query_img is None:
        return False
    
    # Get the base name of the query image for labeling
    query_name = os.path.basename(query_img_path)
    
    # Convert query to grayscale
    query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    
    # Find keypoints and descriptors for query image
    query_keypoints, query_descriptors = feature_extractor.detectAndCompute(query_gray, None)
    
    # Initialize feature matcher
    matcher = create_feature_matcher()
    
    # Match descriptors using knnMatch
    matches = matcher.knnMatch(query_descriptors, target_descriptors, k=2)
    
    # Apply ratio test to filter good matches
    good_matches = filter_matches(matches)
    
    # Check if we have enough matches
    if len(good_matches) < min_match_count:
        print(f"Object '{query_name}': Not enough good matches found - {len(good_matches)}/{min_match_count}")
        return False
    
    # Extract locations of matched keypoints in both images
    query_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    target_pts = np.float32([target_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    
    # Find homography
    H, mask = cv2.findHomography(query_pts, target_pts, cv2.RANSAC, 5.0)
    
    if H is None:
        print(f"Object '{query_name}': Homography could not be computed")
        return False
    
    # Get the corners of the query image
    h, w = query_gray.shape
    corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
    
    # Transform the corners to correspond to the location in the target image
    transformed_corners = cv2.perspectiveTransform(corners, H)
    
    # Validate the homography result
    h, w, _ = target_img.shape
    if not validate_homography(transformed_corners, w, h):
        print(f"Object '{query_name}': Invalid homography result (unreasonable transformation)")
        return False
    
    # Filter out inlier matches using the mask from findHomography
    inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
    
    # Draw the inlier keypoints on the result image
    for match in inlier_matches:
        # Get the keypoint location in the target image
        x, y = target_keypoints[match.trainIdx].pt
        # Draw a circle at each keypoint location
        cv2.circle(result_img, (int(x), int(y)), 3, color, -1)
    
    # Draw a polygon around the detected object
    transformed_corners = transformed_corners.reshape(-1, 2)
    pts = np.array(transformed_corners, dtype=np.int32)
    cv2.polylines(result_img, [pts], True, color, 3)
    
    # Create a semi-transparent highlight overlay
    mask_overlay = np.zeros_like(result_img)
    cv2.fillPoly(mask_overlay, [pts], highlight_color)  # Highlight with object-specific color
    
    # Blend the overlay with the result image
    highlight_alpha = 0.3  # Transparency factor
    cv2.addWeighted(mask_overlay, highlight_alpha, result_img, 1 - highlight_alpha, 0, result_img)
    
    # Add label with object name
    centroid = np.mean(pts, axis=0).astype(int)
    cv2.putText(result_img, query_name, (centroid[0], centroid[1]), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    print(f"Object '{query_name}' detected with {len(inlier_matches)} inlier matches")
    return True