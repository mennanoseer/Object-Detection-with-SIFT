import argparse
import os
import glob
from image_detector import detect_multiple_objects

def main():
    """
    Main function that parses command line arguments and runs object detection.
    """
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect multiple objects using SIFT feature matching')
    parser.add_argument('query_images', nargs='+', help='Paths to query images or a directory of images')
    parser.add_argument('target_image', help='Path to the target image (where to search)')
    parser.add_argument('-o', '--output', help='Path to save the result image', default=None)
    parser.add_argument('-m', '--min-matches', type=int, default=10, 
                        help='Minimum number of good matches required (default: 10)')
    parser.add_argument('-n', '--no-display', action='store_true', 
                        help='Do not display the results')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process query images (handle directory or list of files)
    query_img_paths = []
    for query_path in args.query_images:
        if os.path.isdir(query_path):
            # If it's a directory, get all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
                query_img_paths.extend(glob.glob(os.path.join(query_path, ext)))
        elif os.path.isfile(query_path):
            # If it's a file, add it directly
            query_img_paths.append(query_path)
        else:
            print(f"Warning: '{query_path}' is not a valid file or directory")
    
    if not query_img_paths:
        print("Error: No valid query images found")
        return 2
    
    print(f"Processing {len(query_img_paths)} query images")
    
    # Detect objects
    try:
        _, detection_results = detect_multiple_objects(
            query_img_paths, 
            args.target_image, 
            min_match_count=args.min_matches,
            output_path=args.output,
            display=not args.no_display
        )
        
        # Set exit code based on whether any object was found
        exit_code = 0 if any(detection_results.values()) else 1
        return exit_code
        
    except Exception as e:
        print(f"Error: {e}")
        return 2

if __name__ == "__main__":
    exit(main())