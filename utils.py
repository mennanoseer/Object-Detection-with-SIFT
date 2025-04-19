import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def validate_homography(corners, frame_width, frame_height, min_area_ratio=0.001, max_area_ratio=0.9):
    """
    Validate the homography transformation by checking if the resulting quadrilateral
    has a reasonable size and is not too distorted.
    """
    # Convert to numpy array
    corners = corners.reshape(-1, 2)
    
    # Check if any point is outside the frame with some margin
    margin = 0.2 * max(frame_width, frame_height)  # 20% margin
    for point in corners:
        x, y = point
        if x < -margin or y < -margin or x > frame_width + margin or y > frame_height + margin:
            return False
    
    # Calculate the area of the quadrilateral
    area = cv2.contourArea(np.array(corners, dtype=np.float32))
    frame_area = frame_width * frame_height
    area_ratio = area / frame_area
    
    # Check if area is reasonable (not too small or too large)
    if area_ratio < min_area_ratio or area_ratio > max_area_ratio:
        return False
    
    # Check if the quadrilateral is too stretched
    edges = []
    for i in range(4):
        next_i = (i + 1) % 4
        edge_length = np.sqrt((corners[next_i][0] - corners[i][0])**2 + 
                             (corners[next_i][1] - corners[i][1])**2)
        edges.append(edge_length)
    
    max_edge = max(edges)
    min_edge = min(edges)
    
    # If the ratio of longest to shortest edge is too extreme, reject
    if min_edge == 0 or max_edge / min_edge > 10:
        return False
    
    return True

def display_multiple_results(query_img_paths, target_img_path, result_img):
    """
    Display the query images, target image, and result image with improved layout.
    
    Parameters:
    - query_img_paths: List of paths to query images
    - target_img_path: Path to the target image
    - result_img: Result image with detected objects
    """
    # Read target image
    target_img = cv2.imread(target_img_path)
    if target_img is None:
        print(f"Warning: Could not read target image for display: {target_img_path}")
        return
    target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
    
    # Convert result from BGR to RGB for correct display with matplotlib
    result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
    
    # Filter out invalid query paths
    valid_query_paths = [path for path in query_img_paths if os.path.isfile(path)]
    valid_query_images = []
    
    # Load query images
    for path in valid_query_paths:
        img = cv2.imread(path)
        if img is not None:
            valid_query_images.append((path, cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
    
    num_query = len(valid_query_images)
    
    # Determine the layout
    if num_query <= 3:
        # For small number of query images, use single row layout
        fig, axes = plt.subplots(1, 2 + num_query, figsize=(5 + 3*num_query, 5))
        
        # Display target image
        axes[0].imshow(target_img)
        axes[0].set_title('Target Image')
        axes[0].axis('off')
        
        # Display result image
        axes[1].imshow(result_img)
        axes[1].set_title('Result: Objects Detected')
        axes[1].axis('off')
        
        # Display query images
        for i, (path, img) in enumerate(valid_query_images):
            axes[i+2].imshow(img)
            axes[i+2].set_title(f'Query {i+1}: {os.path.basename(path)}')
            axes[i+2].axis('off')
            
    else:
        # For more query images, use grid layout
        # First row: target and result
        # Subsequent rows: query images (up to 4 per row)
        queries_per_row = 4
        num_query_rows = (num_query + queries_per_row - 1) // queries_per_row
        
        fig = plt.figure(figsize=(12, 5 + 3*num_query_rows))
        
        # Create a grid specification to have control over the layout
        gs = fig.add_gridspec(1 + num_query_rows, queries_per_row)
        
        # Add target image (spans 2 columns)
        ax_target = fig.add_subplot(gs[0, 0:2])
        ax_target.imshow(target_img)
        ax_target.set_title('Target Image')
        ax_target.axis('off')
        
        # Add result image (spans 2 columns)
        ax_result = fig.add_subplot(gs[0, 2:])
        ax_result.imshow(result_img)
        ax_result.set_title('Result: Objects Detected')
        ax_result.axis('off')
        
        # Add query images
        for i, (path, img) in enumerate(valid_query_images):
            row = 1 + (i // queries_per_row)
            col = i % queries_per_row
            ax = fig.add_subplot(gs[row, col])
            ax.imshow(img)
            ax.set_title(f'Query {i+1}: {os.path.basename(path)}', fontsize=10)
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()