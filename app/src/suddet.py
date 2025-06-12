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
        Detect grid lines and structural elements of the Sudoku board with improved robustness

        Args:
            gray_image: Grayscale image

        Returns:
            Tuple of (grid_mask, edges, horizontal_lines, vertical_lines)
        """
        # Multiple preprocessing approaches for different image conditions
        # Standard Gaussian blur
        blurred1 = cv2.GaussianBlur(gray_image, (3, 3), 0)
        # Bilateral filter to preserve edges while reducing noise
        blurred2 = cv2.bilateralFilter(gray_image, 9, 75, 75)
        # Median blur for salt-and-pepper noise
        blurred3 = cv2.medianBlur(gray_image, 3)

        all_edges = []
        
        # Multi-scale edge detection with different blurs
        for blur_img in [blurred1, blurred2, blurred3]:
            # Multiple Canny thresholds for different edge strengths
            edges1 = cv2.Canny(blur_img, 30, 90)
            edges2 = cv2.Canny(blur_img, 50, 150)
            edges3 = cv2.Canny(blur_img, 80, 200)
            edges4 = cv2.Canny(blur_img, 100, 250)
            
            all_edges.extend([edges1, edges2, edges3, edges4])

        # Combine all edge results
        edges_combined = all_edges[0]
        for edge in all_edges[1:]:
            edges_combined = cv2.bitwise_or(edges_combined, edge)

        # Adaptive line detection based on image size
        h, w = gray_image.shape
        min_line_length = min(w, h) // 20  # Minimum line length relative to image size
        max_line_gap = min(w, h) // 40     # Maximum gap in line

        # Detect horizontal lines with multiple kernel sizes
        horizontal_kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length * 2, 1)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_length // 2, 1))
        ]
        
        horizontal_lines_combined = np.zeros_like(edges_combined)
        for kernel in horizontal_kernels:
            h_lines = cv2.morphologyEx(edges_combined, cv2.MORPH_OPEN, kernel)
            horizontal_lines_combined = cv2.bitwise_or(horizontal_lines_combined, h_lines)

        # Detect vertical lines with multiple kernel sizes
        vertical_kernels = [
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length * 2)),
            cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_line_length // 2))
        ]
        
        vertical_lines_combined = np.zeros_like(edges_combined)
        for kernel in vertical_kernels:
            v_lines = cv2.morphologyEx(edges_combined, cv2.MORPH_OPEN, kernel)
            vertical_lines_combined = cv2.bitwise_or(vertical_lines_combined, v_lines)

        # Combine grid lines
        grid_mask = cv2.bitwise_or(horizontal_lines_combined, vertical_lines_combined)

        # Enhanced intersection detection with multiple cross patterns
        cross_kernels = [
            np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8),
            np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=np.uint8),
            np.array([[0, 0, 1, 0, 0], [0, 0, 1, 0, 0], [1, 1, 1, 1, 1], [0, 0, 1, 0, 0], [0, 0, 1, 0, 0]], dtype=np.uint8)
        ]
        
        intersections_combined = np.zeros_like(grid_mask)
        for kernel in cross_kernels:
            intersections = cv2.morphologyEx(grid_mask, cv2.MORPH_OPEN, kernel)
            intersections_combined = cv2.bitwise_or(intersections_combined, intersections)

        # Combine grid with enhanced intersections
        grid_enhanced = cv2.bitwise_or(grid_mask, intersections_combined)
        
        # Clean up the grid mask
        cleanup_kernel = np.ones((2, 2), np.uint8)
        grid_enhanced = cv2.morphologyEx(grid_enhanced, cv2.MORPH_CLOSE, cleanup_kernel)

        return grid_enhanced, edges_combined, horizontal_lines_combined, vertical_lines_combined

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

    def combine_detection_methods(self, image):
        """
        Combine all detection methods for robust board detection

        Args:
            image: Input image (RGB)

        Returns:
            Tuple of detection results
        """
        # height = image.shape[0]
        # width = image.shape[1]
        # image = image[height//50:-height//50,width//50:-width//50]  # Crop to avoid borders
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()

        # Apply all detection methods
        grid_mask, edges, _, _ = self.detect_grid_structure(gray)
        content_mask = self.detect_handwritten_content(gray)

        # Create weighted combination
        h, w = gray.shape
        combined = np.zeros((h, w), dtype=np.float32)

        # Assign weights based on detection confidence
        # Grid structure is important
        combined += grid_mask.astype(np.float32) * 0.30
        # General handwritten content
        combined += content_mask.astype(np.float32) * 0.25
        combined += edges.astype(np.float32) * 0.10          # Edge information

        # Normalize and convert to binary
        combined = np.clip(combined, 0, 255).astype(np.uint8)

        # Apply threshold to create final binary mask
        _, final_mask = cv2.threshold(combined, 60, 255, cv2.THRESH_BINARY)

        # Morphological cleanup
        kernel_close = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(
            final_mask, cv2.MORPH_CLOSE, kernel_close)

        kernel_open = np.ones((3, 3), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel_open)

        return {
            'combined_mask': final_mask,
            'grid_structure': grid_mask,
            'content': content_mask,
            'edges': edges
        }

    def find_board_rectangle(self, mask, image_shape):
        """
        Find the best rectangular contour representing the Sudoku board with enhanced filtering

        Args:
            mask: Binary detection mask
            image_shape: Shape of original image

        Returns:
            Corner points of detected rectangle or None
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Calculate area constraints based on image size
        image_area = image_shape[0] * image_shape[1]
        min_area = image_area * 0.02  # At least 2% of image (more lenient)
        max_area = image_area * 0.85  # At most 85% of image

        # Filter contours by area and geometric properties
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area <= area <= max_area:
                # Additional geometric validation
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 0:
                    # Check circularity (should be low for rectangles)
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    if circularity < 0.85:  # Rectangles have low circularity
                        # Check convexity
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        if hull_area > 0:
                            solidity = area / hull_area
                            if solidity > 0.7:  # Should be reasonably solid
                                valid_contours.append((contour, area, circularity, solidity))

        if not valid_contours:
            return None

        # Sort by a composite score considering area, circularity, and solidity
        def score_contour(contour_data):
            contour, area, circularity, solidity = contour_data
            # Normalize area score (prefer larger contours)
            area_score = area / image_area
            # Prefer lower circularity (more rectangular)
            circ_score = 1.0 - circularity
            # Prefer higher solidity
            solid_score = solidity
            
            return area_score * 0.4 + circ_score * 0.3 + solid_score * 0.3

        valid_contours.sort(key=score_contour, reverse=True)

        # Try to find a good rectangular approximation
        for contour_data in valid_contours:
            contour = contour_data[0]
            
            # Try different epsilon values for polygon approximation
            perimeter = cv2.arcLength(contour, True)

            for epsilon_factor in [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
                epsilon = epsilon_factor * perimeter
                approx = cv2.approxPolyDP(contour, epsilon, True)

                if len(approx) == 4:
                    # Enhanced quadrilateral validation
                    rect = cv2.minAreaRect(contour)
                    width, height = rect[1]

                    if width > 0 and height > 0:
                        aspect_ratio = max(width, height) / min(width, height)

                        # Accept if reasonably square (Sudoku should be close to square)
                        if aspect_ratio <= 2.5:  # More strict aspect ratio
                            # Additional validation: check if corners are well-distributed
                            corners = approx.reshape(4, 2).astype(np.float32)
                            
                            # Check if it's not a degenerate quadrilateral
                            area_quad = cv2.contourArea(approx)
                            if area_quad > min_area:
                                # Validate corner angles (should be close to 90 degrees for Sudoku)
                                if self._validate_corner_angles(corners):
                                    return approx

        # Enhanced fallback: try to find the most square-like contour
        if valid_contours:
            best_contour = None
            best_score = 0
            
            for contour_data in valid_contours[:3]:  # Check top 3 candidates
                contour = contour_data[0]
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                
                if width > 0 and height > 0:
                    aspect_ratio = max(width, height) / min(width, height)
                    # Score based on how close to square it is
                    square_score = 1.0 / aspect_ratio
                    area_score = contour_data[1] / image_area
                    
                    total_score = square_score * 0.7 + area_score * 0.3
                    
                    if total_score > best_score:
                        best_score = total_score
                        best_contour = contour

            if best_contour is not None:
                # Use oriented bounding rectangle for better results
                rect = cv2.minAreaRect(best_contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                
                # Convert to the expected format
                rectangle_points = box.reshape(-1, 1, 2)
                return rectangle_points

        return None
    
    def _validate_corner_angles(self, corners):
        """
        Validate that corner angles are reasonably close to 90 degrees
        
        Args:
            corners: Array of 4 corner points
            
        Returns:
            True if angles are valid for a Sudoku board
        """
        def angle_between_vectors(v1, v2):
            """Calculate angle between two vectors in degrees"""
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            # Clamp to avoid numerical errors
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            return np.degrees(np.arccos(cos_angle))
        
        angles = []
        for i in range(4):
            # Get three consecutive points
            p1 = corners[i]
            p2 = corners[(i + 1) % 4]
            p3 = corners[(i + 2) % 4]
            
            # Create vectors
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Calculate angle
            angle = angle_between_vectors(v1, v2)
            angles.append(angle)
        
        # Check if all angles are reasonably close to 90 degrees
        for angle in angles:
            if not (60 <= angle <= 120):  # Allow some tolerance
                return False
        
        return True

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
            # Perform detection
            detection_results = self.combine_detection_methods(image)

            # Find board rectangle
            corners = self.find_board_rectangle(
                detection_results['combined_mask'], image.shape)

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

    def visualize_detection(self, result, save_path=None, figsize=(16, 12)):
        """
        Visualize the detection process and results

        Args:
            result: Result dictionary from detect_and_extract
            save_path: Optional path to save visualization
            figsize: Figure size for matplotlib
        """
        if not result.get('original_image') is not None:
            print("No image data to visualize")
            return

        _, axes = plt.subplots(4, 2, figsize=figsize)

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

            axes[1, 0].imshow(masks['content'], cmap='gray')
            axes[1, 0].set_title('Content Detection')
            axes[1, 0].axis('off')

            axes[1, 1].imshow(masks['edges'], cmap='gray')
            axes[1, 1].set_title('Edge Detection')
            axes[1, 1].axis('off')

            axes[2, 0].imshow(masks['combined_mask'], cmap='gray')
            axes[2, 0].set_title('Combined Mask')
            axes[2, 0].axis('off')

        # Detected corners visualization
        corner_vis = result['original_image'].copy()
        if result['corners'] is not None:
            cv2.drawContours(
                corner_vis, [result['corners']], -1, (255, 0, 0), 3)
            # Draw corner points
            for point in result['corners'].reshape(-1, 2):
                cv2.circle(corner_vis, tuple(
                    point.astype(int)), 8, (0, 255, 0), -1)

        axes[2, 1].imshow(corner_vis)
        axes[2, 1].set_title('Detected Board Corners')
        axes[2, 1].axis('off')

        # Final extracted board
        if result['board'] is not None:
            axes[3, 0].imshow(result['board'])
            axes[3, 0].set_title('Extracted Sudoku Board')
        else:
            axes[3, 0].text(0.5, 0.5, f"Extraction Failed\\n{result.get('error', 'Unknown error')}",
                            ha='center', va='center', transform=axes[2, 1].transAxes, fontsize=12)
            axes[3, 0].set_title('Extraction Result')
        axes[3, 0].axis('off')

        # Remove empty subplot
        axes[3, 1].axis('off')

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
    test_dir = Path("/home/tomas/PYT/motustom/app/tests/images")

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
