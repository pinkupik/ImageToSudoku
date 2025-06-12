# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Comprehensive Sudoku Board Detection and Extraction Module

This module provides robust detection and extraction of Sudoku boards from images,
with special optimization for handwritten content including blue pen writing.

Features:
- Multiple detection methods for different image types
- Grid structure detection for traditional printed Sudoku
- Adaptive content detection for handwritten digits
- Perspective correction for angled boards
- Robust rectangle detection with fallback options

Usage:
    from app.src.suddet import SudokuBoardDetector
    
    detector = SudokuBoardDetector()
    result = detector.detect_and_extract('path/to/sudoku_image.jpg')
    
    if result['success']:
        extracted_board = result['board']
        # Use the extracted board for further processing
"""
import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# @generated "[partially]" Copilot Claude Sonnet 4: Repair this class so it correctly detects and extracts Sudoku boards. Add nice visualizations.
class SudokuBoardDetector:
    """
    Advanced Sudoku board detector optimized for various image types
    including handwritten content
    """

    def __init__(self, output_size=450, debug=False):
        """Initialize the detector

        Args:
            output_size (int): Size of the extracted square board
            debug (bool): Enable debug visualizations (unused in production)
        """
        self.output_size = output_size
        self.debug = debug

    def detect_grid_structure(self, gray_image):
        """
        Detect grid lines and structural elements of the Sudoku board using
        advanced line detection and Hough transforms

        Args:
            gray_image: Grayscale image

        Returns:
            Tuple of (grid_mask, edges, horizontal_lines, vertical_lines, hough_lines)
        """
        # Preprocessing with multiple blur levels
        blurred = cv2.GaussianBlur(gray_image, (3, 3), 0)
        
        # Edge detection with multiple parameters
        edges1 = cv2.Canny(blurred, 20, 80)
        edges2 = cv2.Canny(blurred, 50, 150)
        edges3 = cv2.Canny(blurred, 80, 200)

        # Combine edge results
        edges_combined = cv2.bitwise_or(edges1, edges2)
        edges_combined = cv2.bitwise_or(edges_combined, edges3)

        # Hough Line Detection for more accurate line detection
        hough_lines = cv2.HoughLinesP(edges_combined, 1, np.pi/180, 
                                     threshold=50, minLineLength=30, maxLineGap=10)
        
        # Create line mask from Hough lines
        h, w = gray_image.shape
        hough_mask = np.zeros((h, w), dtype=np.uint8)
        
        horizontal_hough = np.zeros((h, w), dtype=np.uint8)
        vertical_hough = np.zeros((h, w), dtype=np.uint8)
        
        if hough_lines is not None:
            for line in hough_lines:
                x1, y1, x2, y2 = line[0]
                # Calculate line angle
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                # Filter for horizontal and vertical lines
                if abs(angle) < 15 or abs(angle) > 165:  # Horizontal lines
                    cv2.line(horizontal_hough, (x1, y1), (x2, y2), 255, 2)
                    cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)
                elif 75 < abs(angle) < 105:  # Vertical lines
                    cv2.line(vertical_hough, (x1, y1), (x2, y2), 255, 2)
                    cv2.line(hough_mask, (x1, y1), (x2, y2), 255, 2)

        # Morphological line detection (fallback)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        horizontal_lines = cv2.morphologyEx(
            edges_combined, cv2.MORPH_OPEN, horizontal_kernel)

        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        vertical_lines = cv2.morphologyEx(
            edges_combined, cv2.MORPH_OPEN, vertical_kernel)

        # Combine Hough and morphological results
        horizontal_combined = cv2.bitwise_or(horizontal_lines, horizontal_hough)
        vertical_combined = cv2.bitwise_or(vertical_lines, vertical_hough)
        
        # Combine grid lines
        grid_mask = cv2.bitwise_or(horizontal_combined, vertical_combined)

        # Enhance grid intersections with better kernel
        kernel_cross = np.array([
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0]
        ], dtype=np.uint8)
        intersections = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, kernel_cross)

        # Add intersection enhancement
        kernel_plus = np.array([
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
        ], dtype=np.uint8)
        intersections_plus = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, kernel_plus)
        
        # Combine all intersection detections
        intersections_final = cv2.bitwise_or(intersections, intersections_plus)
        
        # Combine grid with intersections
        grid_enhanced = cv2.bitwise_or(grid_mask, intersections_final)

        return grid_enhanced, edges_combined, horizontal_combined, vertical_combined, hough_lines

    def detect_handwritten_content(self, gray_image):
        """
        Detect handwritten content using adaptive thresholding methods

        Args:
            gray_image: Grayscale image

        Returns:
            Binary mask of detected content
        """
        # Multiple adaptive thresholding approaches
        methods = [
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 11, 2),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 15, 3),
            (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 19, 4),
            (cv2.ADAPTIVE_THRESH_MEAN_C, 11, 2),
            (cv2.ADAPTIVE_THRESH_MEAN_C, 15, 3),
        ]

        adaptive_results = []
        for method, block_size, c in methods:
            result = cv2.adaptiveThreshold(
                gray_image, 255, method, cv2.THRESH_BINARY_INV, block_size, c)
            adaptive_results.append(result)

        # Combine adaptive threshold results
        combined_adaptive = adaptive_results[0]
        for result in adaptive_results[1:]:
            combined_adaptive = cv2.bitwise_or(combined_adaptive, result)

        # Remove very small noise
        kernel_noise = np.ones((2, 2), np.uint8)
        combined_adaptive = cv2.morphologyEx(
            combined_adaptive, cv2.MORPH_OPEN, kernel_noise)

        return combined_adaptive

    def preprocess_image(self, image):
        """
        Preprocess the image to enhance Sudoku detection
        
        Args:
            image: Input image (RGB)
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise
        denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
        
        # Convert back to RGB if original was RGB
        if len(image.shape) == 3:
            processed = cv2.cvtColor(denoised, cv2.COLOR_GRAY2RGB)
        else:
            processed = denoised
            
        return processed

    def apply_center_focus_filter(self, mask, image_shape, focus_factor=0.7):
        """
        Apply a center-focused filter to emphasize objects in the middle of the image
        
        Args:
            mask: Binary detection mask
            image_shape: Shape of the image
            focus_factor: How much to emphasize center (0.0-1.0)
            
        Returns:
            Filtered mask with center emphasis
        """
        h, w = image_shape[:2]
        
        # Create a circular gradient mask favoring the center
        center_x, center_y = w // 2, h // 2
        max_radius = min(center_x, center_y)
        
        y, x = np.ogrid[:h, :w]
        distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
        
        # Normalize distances to 0-1 range
        normalized_distances = distances / max_radius
        normalized_distances = np.clip(normalized_distances, 0, 1)
        
        # Create weight mask (higher weight at center)
        center_weights = 1.0 - (normalized_distances * focus_factor)
        center_weights = np.clip(center_weights, 1.0 - focus_factor, 1.0)
        
        # Apply weights to mask
        filtered_mask = (mask.astype(np.float32) * center_weights).astype(np.uint8)
        
        return filtered_mask

    def combine_detection_methods(self, image):
        """
        Combine all detection methods for robust board detection

        Args:
            image: Input image (RGB)

        Returns:
            Tuple of detection results
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply all detection methods
        grid_mask, edges, horizontal_lines, vertical_lines, hough_lines = self.detect_grid_structure(gray)
        content_mask = self.detect_handwritten_content(gray)

        # Create weighted combination
        h, w = gray.shape
        combined = np.zeros((h, w), dtype=np.float32)

        # Assign weights based on detection confidence
        combined += grid_mask.astype(np.float32) * 0.40       # Grid structure is most important
        combined += content_mask.astype(np.float32) * 0.25    # Handwritten content
        combined += edges.astype(np.float32) * 0.15           # Edge information
        combined += horizontal_lines.astype(np.float32) * 0.10 # Horizontal lines
        combined += vertical_lines.astype(np.float32) * 0.10   # Vertical lines

        # Normalize and convert to binary
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # Apply threshold to create final binary mask
        _, final_mask = cv2.threshold(combined, 80, 255, cv2.THRESH_BINARY)

        # Apply center focus filter to prioritize objects in the middle
        final_mask = self.apply_center_focus_filter(final_mask, gray.shape, focus_factor=0.6)

        # Morphological cleanup
        kernel_close = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)

        return {
            'combined_mask': final_mask,
            'grid_structure': grid_mask,
            'content': content_mask,
            'edges': edges,
            'horizontal_lines': horizontal_lines,
            'vertical_lines': vertical_lines,
            'hough_lines': hough_lines
        }

    def calculate_center_score(self, contour, image_shape):
        """
        Calculate how close a contour is to the center of the image.
        Higher scores for contours closer to center.
        
        Args:
            contour: OpenCV contour
            image_shape: Shape of the image (height, width)
            
        Returns:
            Score between 0 and 1, where 1 is perfect center
        """
        h, w = image_shape[:2]
        image_center = np.array([w/2, h/2])
        
        # Get contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0
        
        contour_center = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]])
        
        # Calculate distance from center
        max_distance = np.sqrt((w/2)**2 + (h/2)**2)
        distance = np.linalg.norm(contour_center - image_center)
        
        # Convert to score (closer = higher score)
        center_score = 1 - (distance / max_distance)
        
        # Apply exponential weighting to heavily favor center objects
        center_score = center_score ** 2
        
        return center_score

    def validate_sudoku_grid(self, corners, hough_lines, image_shape):
        """
        Validate if the detected rectangle looks like a Sudoku grid
        by checking for internal grid lines
        
        Args:
            corners: Corner points of detected rectangle
            hough_lines: Detected Hough lines
            image_shape: Shape of the image
            
        Returns:
            Validation score between 0 and 1
        """
        if hough_lines is None or len(hough_lines) < 8:
            return 0.1  # Need at least some lines for a grid
        
        # Get bounding box of corners
        x_coords = corners.reshape(-1, 2)[:, 0]
        y_coords = corners.reshape(-1, 2)[:, 1]
        
        min_x, max_x = np.min(x_coords), np.max(x_coords)
        min_y, max_y = np.min(y_coords), np.max(y_coords)
        
        # Count lines within the detected rectangle
        internal_horizontal = 0
        internal_vertical = 0
        
        for line in hough_lines:
            x1, y1, x2, y2 = line[0]
            
            # Check if line is mostly within the rectangle
            points_in_rect = 0
            for px, py in [(x1, y1), (x2, y2)]:
                if min_x <= px <= max_x and min_y <= py <= max_y:
                    points_in_rect += 1
            
            if points_in_rect >= 1:  # At least one endpoint in rectangle
                # Calculate line angle
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                
                if abs(angle) < 15 or abs(angle) > 165:  # Horizontal
                    internal_horizontal += 1
                elif 75 < abs(angle) < 105:  # Vertical
                    internal_vertical += 1
        
        # Sudoku should have multiple internal lines in both directions
        horizontal_score = min(internal_horizontal / 6, 1.0)  # At least 6 horizontal lines
        vertical_score = min(internal_vertical / 6, 1.0)      # At least 6 vertical lines
        
        return (horizontal_score + vertical_score) / 2

    def find_board_rectangle(self, mask, image_shape, detection_results=None):
        """
        Find the best rectangular contour representing the Sudoku board
        with center-focused filtering and grid validation

        Args:
            mask: Binary detection mask
            image_shape: Shape of original image
            detection_results: Additional detection information

        Returns:
            Corner points of detected rectangle or None
        """
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Calculate area constraints
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * 0.05  # At least 5% of image (increased from 3%)
        max_area = image_area * 0.85  # At most 85% of image (decreased from 90%)

        # Filter and score contours
        candidate_contours = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Calculate center score
                center_score = self.calculate_center_score(contour, image_shape)
                
                # Calculate area score (prefer larger areas, but not too large)
                area_ratio = area / image_area
                if area_ratio < 0.1:
                    area_score = area_ratio / 0.1  # Linear increase up to 10%
                else:
                    area_score = 1.0 - (area_ratio - 0.1) / 0.5  # Decrease after 10%
                area_score = max(0, min(1, area_score))
                
                # Calculate perimeter regularity (squares have specific perimeter/area ratio)
                perimeter = cv2.arcLength(contour, True)
                if area > 0:
                    regularity = (perimeter ** 2) / (16 * area)  # Perfect square = 1
                    regularity_score = 1.0 / (1.0 + abs(regularity - 1.0))
                else:
                    regularity_score = 0
                
                # Combined score with heavy weighting on center position
                total_score = (center_score * 0.5 +      # 50% weight on center position
                             area_score * 0.25 +         # 25% weight on area
                             regularity_score * 0.25)    # 25% weight on shape regularity
                
                candidate_contours.append((contour, total_score))

        if not candidate_contours:
            return None

        # Sort by total score (highest first)
        candidate_contours.sort(key=lambda x: x[1], reverse=True)

        # Try to find a good rectangular approximation from top candidates
        for contour, score in candidate_contours[:5]:  # Check top 5 candidates
            perimeter = cv2.arcLength(contour, True)

            for epsilon_factor in [0.01, 0.02, 0.03, 0.05, 0.08, 0.10]:
                epsilon = epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    # Check if the quadrilateral is reasonably square
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]

                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)

                        # Accept if reasonably square (tighter tolerance)
                        if aspect_ratio <= 2.0:  # Reduced from 3.0
                            # Additional validation using grid lines
                            if detection_results and 'hough_lines' in detection_results:
                                grid_score = self.validate_sudoku_grid(
                                    approx, detection_results['hough_lines'], image_shape)
                                
                                # Require minimum grid validation for acceptance
                                if grid_score > 0.2:  # At least some internal structure
                                    return approx
                            else:
                                return approx

        # Fallback: use bounding rectangle of highest-scoring contour
        if candidate_contours:
            best_contour = candidate_contours[0][0]
            x, y, w, h = cv2.boundingRect(best_contour)

            # Only use fallback if it's reasonably square and centered
            aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else float('inf')
            center_score = self.calculate_center_score(best_contour, image_shape)
            
            if aspect_ratio <= 2.5 and center_score > 0.3:  # Reasonably square and somewhat centered
                rectangle_points = np.array([
                    [[x, y]],
                    [[x + w, y]],
                    [[x + w, y + h]],
                    [[x, y + h]]
                ], dtype=np.int32)

                return rectangle_points

        return None

    def order_corner_points(self, corners):
        """
        Order corner points in consistent order: top-left, top-right, bottom-right, bottom-left

        Args:
            corners: Array of 4 corner points

        Returns:
            Ordered corner points
        """
        # Reshape to (4, 2)
        points = corners.reshape(4, 2).astype(np.float32)

        # Initialize ordered rectangle
        ordered = np.zeros((4, 2), dtype=np.float32)

        # Sum of coordinates to find top-left (minimum) and bottom-right (maximum)
        point_sums = points.sum(axis=1)
        ordered[0] = points[np.argmin(point_sums)]  # top-left
        ordered[2] = points[np.argmax(point_sums)]  # bottom-right

        # Difference of coordinates to find top-right and bottom-left
        point_diffs = np.diff(points, axis=1)
        ordered[1] = points[np.argmin(point_diffs)]  # top-right
        ordered[3] = points[np.argmax(point_diffs)]  # bottom-left

        return ordered

    def extract_board_perspective(self, image, corners):
        """
        Extract the Sudoku board using perspective transformation

        Args:
            image: Original image
            corners: Corner points of the board

        Returns:
            Warped board image
        """
        # Order the corner points
        ordered_corners = self.order_corner_points(corners)

        # Define destination points for a square output
        dst_points = np.array([
            [0, 0],
            [self.output_size - 1, 0],
            [self.output_size - 1, self.output_size - 1],
            [0, self.output_size - 1]
        ], dtype=np.float32)

        # Calculate perspective transformation matrix
        transform_matrix = cv2.getPerspectiveTransform(
            ordered_corners, dst_points)

        # Apply perspective transformation
        warped_image = cv2.warpPerspective(image, transform_matrix,
                                           (self.output_size, self.output_size))

        return warped_image

    def detect_and_extract(self, image_path_or_array):
        """
        Main method to detect and extract Sudoku board

        Args:
            image_path_or_array: Path to image file or image array

        Returns:
            Dictionary with detection results
        """
        # Load image
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                return {'success': False, 'error': 'Image file not found'}
            image = cv2.imread(image_path_or_array)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if image is None:
                return {'success': False, 'error': 'Could not load image'}
        else:
            image = image_path_or_array.copy()

        try:
            # Preprocess image for better detection
            preprocessed_image = self.preprocess_image(image)
            
            # Perform detection
            detection_results = self.combine_detection_methods(preprocessed_image)

            # Find board rectangle with improved center-focused and grid-validated detection
            corners = self.find_board_rectangle(
                detection_results['combined_mask'], image.shape, detection_results)

            result = {
                'success': False,
                'original_image': image,
                'detection_masks': detection_results,
                'corners': corners,
                'board': None,
                'error': None
            }

            # Extract board if corners found
            if corners is not None:
                try:
                    warped_board = self.extract_board_perspective(
                        image, corners)
                    result['board'] = warped_board
                    result['success'] = True
                except Exception as e:
                    result['error'] = f'Perspective transformation failed: {str(e)}'
                    result['board'] = image
            else:
                result['error'] = 'No valid board rectangle detected'
                result['board'] = image

            return result

        except Exception as e:
            return {'success': False, 'error': f'Detection failed: {str(e)}', 'board': image}

    def visualize_detection(self, result, save_path=None, figsize=(18, 14)):
        """
        Visualize the detection process and results with enhanced visualizations

        Args:
            result: Result dictionary from detect_and_extract
            save_path: Optional path to save visualization
            figsize: Figure size for matplotlib
        """
        if not result.get('original_image') is not None:
            print("No image data to visualize")
            return

        _, axes = plt.subplots(4, 3, figsize=figsize)

        # Original image
        axes[0, 0].imshow(result['original_image'])
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        # Detection masks
        if 'detection_masks' in result:
            masks = result['detection_masks']

            axes[0, 1].imshow(masks['grid_structure'], cmap='gray')
            axes[0, 1].set_title('Grid Structure')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(masks['content'], cmap='gray')
            axes[0, 2].set_title('Content Detection')
            axes[0, 2].axis('off')

            axes[1, 0].imshow(masks['edges'], cmap='gray')
            axes[1, 0].set_title('Edge Detection')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(masks['horizontal_lines'], cmap='gray')
            axes[1, 1].set_title('Horizontal Lines')
            axes[1, 1].axis('off')

            axes[1, 2].imshow(masks['vertical_lines'], cmap='gray')
            axes[1, 2].set_title('Vertical Lines')
            axes[1, 2].axis('off')

            axes[2, 0].imshow(masks['combined_mask'], cmap='gray')
            axes[2, 0].set_title('Combined Mask')
            axes[2, 0].axis('off')

        # Hough lines visualization
        hough_vis = result['original_image'].copy()
        if 'detection_masks' in result and 'hough_lines' in result['detection_masks']:
            hough_lines = result['detection_masks']['hough_lines']
            if hough_lines is not None:
                for line in hough_lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(hough_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        axes[2, 1].imshow(hough_vis)
        axes[2, 1].set_title('Detected Hough Lines')
        axes[2, 1].axis('off')

        # Detected corners visualization
        corner_vis = result['original_image'].copy()
        if result['corners'] is not None:
            cv2.drawContours(corner_vis, [result['corners']], -1, (255, 0, 0), 3)
            # Draw corner points
            for point in result['corners'].reshape(-1, 2):
                cv2.circle(corner_vis, tuple(point.astype(int)), 8, (0, 255, 0), -1)

        axes[2, 2].imshow(corner_vis)
        axes[2, 2].set_title('Detected Board Corners')
        axes[2, 2].axis('off')

        # Final extracted board
        if result['board'] is not None:
            axes[3, 0].imshow(result['board'])
            axes[3, 0].set_title('Extracted Sudoku Board')
        else:
            axes[3, 0].text(0.5, 0.5, f"Extraction Failed\n{result.get('error', 'Unknown error')}",
                            ha='center', va='center', transform=axes[3, 0].transAxes, fontsize=12)
            axes[3, 0].set_title('Extraction Result')
        axes[3, 0].axis('off')

        # Remove empty subplots
        axes[3, 1].axis('off')
        axes[3, 2].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()


def demo_sudoku_detection():
    """
    Demonstration function to test the Sudoku detector
    """
    # Initialize detector
    detector = SudokuBoardDetector(output_size=450, debug=True)

    # Test directory
    test_dir = Path("/home/tomas/random/ImageToSudoku/app/tests/images")

    if not test_dir.exists():
        print(f"Test directory {test_dir} not found")
        return

    # Test images
    test_images = ["image1.jpg", "image2.png", "image3.png", "image4.png", "image5.jpg", "randomroate.jpg"]

    print("=== Sudoku Board Detection Demo ===")
    print(f"Output board size: {detector.output_size}x{detector.output_size}")
    print()

    for img_name in test_images:
        img_path = test_dir / img_name

        if img_path.exists():
            print(f"Processing: {img_name}")

            # Detect and extract
            result = detector.detect_and_extract(str(img_path))

            if result['success']:
                print("  ✓ Successfully extracted Sudoku board")
                print(f"  Board shape: {result['board'].shape}")
            else:
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")

            # Visualize results
            detector.visualize_detection(result)

        else:
            print(f"Image not found: {img_name}")

        print()

    print("Demo completed!")


if __name__ == "__main__":
    demo_sudoku_detection()
