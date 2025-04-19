import cv2
import numpy as np
import os

def load_image(image_path):
    """
    Load an image from a file path.
    
    Parameters:
    - image_path: Path to the image
    
    Returns:
    - Image as numpy array, or None if the image could not be loaded
    """
    if not os.path.isfile(image_path):
        print(f"Image not found: {image_path}")
        return None
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
    
    return image

def create_feature_extractor():
    """
    Create a SIFT feature extractor.
    
    Returns:
    - SIFT feature extractor instance
    """
    return cv2.SIFT_create()

def create_feature_matcher():
    """
    Create a FLANN-based matcher for SIFT features.
    
    Returns:
    - Feature matcher instance
    """
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)

def filter_matches(matches, ratio=0.7):
    """
    Apply ratio test to filter good matches.
    
    Parameters:
    - matches: Matches from knnMatch
    - ratio: Ratio threshold for filtering
    
    Returns:
    - List of good matches
    """
    good_matches = []
    for m, n in matches:
        if m.distance < ratio * n.distance:
            good_matches.append(m)
    return good_matches