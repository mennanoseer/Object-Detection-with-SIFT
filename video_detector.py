import cv2
import numpy as np
from utils import validate_homography

def _detect_object_in_frame(matcher, query_keypoints, query_descriptors, frame_keypoints, 
                           frame_descriptors, query_gray, result_frame, min_match_count, 
                           ratio_threshold, ransac_threshold, frame_width, frame_height,
                           last_valid_H, object_missed_count):
    """Detect object in current frame and return confidence score."""
    object_found = False
    confidence_score = 0.0
    updated_H = None
    
    # Match descriptors
    matches = matcher.knnMatch(query_descriptors, frame_descriptors, k=2)
    
    # Apply ratio test
    good_matches = []
    for match in matches:
        if len(match) == 2:  # Ensure we have two matches for the ratio test
            m, n = match
            if m.distance < ratio_threshold * n.distance:
                good_matches.append(m)
    
    # Calculate initial confidence based on number of good matches
    initial_confidence = min(1.0, len(good_matches) / (3 * min_match_count)) * 100
    
    # Process matches if enough are found
    if len(good_matches) >= min_match_count:
        # Extract locations of matched keypoints in both images
        query_pts = np.float32([query_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        frame_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(query_pts, frame_pts, cv2.RANSAC, ransac_threshold)
        
        if H is not None:
            # Get the corners of the query image
            h, w = query_gray.shape
            corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            # Transform the corners to correspond to the location in the frame
            transformed_corners = cv2.perspectiveTransform(corners, H)
            
            # Validate the homography
            is_valid = validate_homography(transformed_corners, frame_width, frame_height)
            
            if is_valid:
                object_found = True
                updated_H = H
                
                # Extract inlier matches
                inlier_matches = [good_matches[i] for i in range(len(good_matches)) if mask[i]]
                inlier_ratio = len(inlier_matches) / len(good_matches) if good_matches else 0
                
                # Calculate confidence score
                match_weight = 0.7
                inlier_weight = 0.3
                match_confidence = min(1.0, len(inlier_matches) / (2 * min_match_count)) * 100
                inlier_confidence = inlier_ratio * 100
                confidence_score = (match_weight * match_confidence) + (inlier_weight * inlier_confidence)
                
                # Draw the inlier keypoints and highlight the object
                _draw_object_detection(result_frame, frame_keypoints, inlier_matches, 
                                      transformed_corners, confidence_score)
            else:
                # Use last valid homography if available
                confidence_score = _handle_invalid_homography(result_frame, last_valid_H, query_gray,
                                                           object_missed_count, initial_confidence)
        else:
            confidence_score = initial_confidence * 0.4  # 40% of initial confidence
            cv2.putText(result_frame, "Homography estimation failed", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result_frame, f"Confidence: {confidence_score:.1f}%", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        # Use last valid homography if available
        confidence_score = _handle_not_enough_matches(result_frame, last_valid_H, query_gray,
                                                   good_matches, min_match_count, 
                                                   object_missed_count, initial_confidence)
    
    return object_found, confidence_score, updated_H


def _draw_object_detection(result_frame, frame_keypoints, inlier_matches, transformed_corners, confidence_score):
    """Draw the detected object on the result frame."""
    # Draw the inlier keypoints
    for match in inlier_matches:
        x, y = frame_keypoints[match.trainIdx].pt
        cv2.circle(result_frame, (int(x), int(y)), 3, (0, 255, 0), -1)
    
    # Draw a polygon around the detected object
    transformed_corners = transformed_corners.reshape(-1, 2)
    pts = np.array(transformed_corners, dtype=np.int32)
    cv2.polylines(result_frame, [pts], True, (0, 0, 255), 3)
    
    # Create a semi-transparent highlight overlay
    overlay = result_frame.copy()
    mask_img = np.zeros_like(result_frame)
    cv2.fillPoly(mask_img, [pts], (0, 255, 255))  # Yellow highlight
    
    # Blend the overlay with the original image
    highlight_alpha = 0.3  # Transparency factor
    cv2.addWeighted(mask_img, highlight_alpha, result_frame, 1 - highlight_alpha, 0, result_frame)
    
    # Add text indicating object found with confidence
    cv2.putText(result_frame, f"Object Found: {len(inlier_matches)} matches", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(result_frame, f"Confidence: {confidence_score:.1f}%", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)


def _handle_invalid_homography(result_frame, last_valid_H, query_gray,
                              object_missed_count, initial_confidence):
    """Handle case when homography is invalid but we have a previous valid one."""
    if last_valid_H is not None and object_missed_count < 10:
        h, w = query_gray.shape
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        transformed_corners = cv2.perspectiveTransform(corners, last_valid_H)
        transformed_corners = transformed_corners.reshape(-1, 2)
        pts = np.array(transformed_corners, dtype=np.int32)
        
        # Draw with a different color to indicate prediction
        cv2.polylines(result_frame, [pts], True, (255, 0, 0), 2)  # Blue outline for prediction
        
        # Add text indicating predicted location with reduced confidence
        confidence_score = max(0, 50 - (object_missed_count * 5))  # Decaying confidence
        cv2.putText(result_frame, "Predicted location", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(result_frame, f"Confidence: {confidence_score:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        confidence_score = initial_confidence * 0.5  # Half the initial confidence
        cv2.putText(result_frame, "Invalid homography", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Confidence: {confidence_score:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return confidence_score


def _handle_not_enough_matches(result_frame, last_valid_H, query_gray, good_matches, 
                              min_match_count, object_missed_count, initial_confidence):
    """Handle case when there are not enough matches."""
    if last_valid_H is not None and object_missed_count < 10:
        h, w = query_gray.shape
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        transformed_corners = cv2.perspectiveTransform(corners, last_valid_H)
        transformed_corners = transformed_corners.reshape(-1, 2)
        pts = np.array(transformed_corners, dtype=np.int32)
        
        # Draw with a different color to indicate prediction
        cv2.polylines(result_frame, [pts], True, (255, 0, 0), 2)  # Blue outline for prediction
        
        # Add text indicating predicted location with reduced confidence
        confidence_score = max(0, 50 - (object_missed_count * 5))  # Decaying confidence
        cv2.putText(result_frame, "Predicted location", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(result_frame, f"Confidence: {confidence_score:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    else:
        confidence_score = initial_confidence
        cv2.putText(result_frame, f"Not enough matches: {len(good_matches)}/{min_match_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(result_frame, f"Confidence: {confidence_score:.1f}%", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return confidence_score


def _handle_no_features_frame(result_frame, window_name, out):
    """Handle frame with no features detected."""
    cv2.putText(result_frame, "No features in frame", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow(window_name, result_frame)
    if out is not None:
        out.write(result_frame)


def _check_quit():
    """Check if user pressed 'q' to quit."""
    key = cv2.waitKey(1) & 0xFF
    return key == ord('q')


def _display_progress_info(result_frame, frame_count, total_frames, fps_display, detect_count):
    """Display progress information on the frame."""
    cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(result_frame, f"FPS: {fps_display:.1f}", (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    detection_rate = (detect_count/frame_count*100) if frame_count > 0 else 0
    cv2.putText(result_frame, f"Detection rate: {detection_rate:.1f}%", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)