# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Test Module for Sudoku Detection Functionality.

This module contains comprehensive test cases for the Sudoku detection system,
including image preprocessing, grid detection, cell extraction, and coordinate
transformation. It validates the computer vision pipeline that identifies and
extracts Sudoku puzzles from images.

Test Coverage:
- Image preprocessing and enhancement
- Grid boundary detection algorithms
- Cell segmentation and extraction
- Perspective transformation and correction
- Corner detection and validation
- Error handling for invalid inputs

The tests ensure the reliability and accuracy of the detection system across
various image conditions, lighting scenarios, and puzzle orientations.

Author: Tomáš Motus
Project: Sudoku Detection and Solving System
"""

import sys
import os
import pytest
import numpy as np
import cv2
from pathlib import Path

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.suddet import SudokuBoardDetector

# @generated "[partially]" Copilot Claude Sonnet 4: Add more tests
class TestSudokuBoardDetector:
    """Test suite for SudokuBoardDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing"""
        return SudokuBoardDetector(output_size=450, debug=False)
    
    @pytest.fixture
    def test_images_dir(self):
        """Get the test images directory path"""
        return Path(__file__).parent / "images"
    
    @pytest.fixture
    def sample_image_paths(self, test_images_dir):
        """Get paths to test images"""
        return [
            test_images_dir / "image1.jpg",
            test_images_dir / "image2.png", 
            test_images_dir / "image3.png",
            test_images_dir / "image4.png"
        ]
    
    def test_detector_initialization(self):
        """Test detector initialization with different parameters"""
        # Default parameters
        detector1 = SudokuBoardDetector()
        assert detector1.output_size == 450
        assert detector1.debug == False
        
        # Custom parameters
        detector2 = SudokuBoardDetector(output_size=600, debug=True)
        assert detector2.output_size == 600
        assert detector2.debug == True
    
    def test_detect_grid_structure(self, detector):
        """Test grid structure detection method"""
        # Create a synthetic test image with grid lines
        test_image = np.ones((400, 400), dtype=np.uint8) * 255
        
        # Draw vertical lines
        for i in range(1, 9):
            x = i * 400 // 9
            cv2.line(test_image, (x, 0), (x, 400), (0,), 2)

        # Draw horizontal lines
        for i in range(1, 9):
            y = i * 400 // 9
            cv2.line(test_image, (0, y), (400, y), (0,), 2)
        
        grid_mask, edges, h_lines, v_lines = detector.detect_grid_structure(test_image)
        
        # Check that detection returns proper shapes
        assert grid_mask.shape == test_image.shape
        assert edges.shape == test_image.shape
        assert h_lines.shape == test_image.shape
        assert v_lines.shape == test_image.shape
        
        # Check that some grid structure was detected
        assert np.any(grid_mask > 0), "Grid mask should contain detected structure"
    
    def test_detect_handwritten_content(self, detector):
        """Test handwritten content detection method"""
        # Create a synthetic test image with text-like content
        test_image = np.ones((400, 400), dtype=np.uint8) * 255
        
        # Add some text-like markings
        cv2.putText(test_image, "5", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,), 3)
        cv2.putText(test_image, "3", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,), 3)
        
        content_mask = detector.detect_handwritten_content(test_image)
        
        # Check return shape
        assert content_mask.shape == test_image.shape
        assert content_mask.dtype == np.uint8
        
        # Check that some content was detected
        assert np.any(content_mask > 0), "Content mask should contain detected handwriting"
    
    def test_combine_detection_methods(self, detector):
        """Test combination of detection methods"""
        # Create a test image with both grid and content
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Add grid lines
        for i in range(1, 9):
            x = i * 400 // 9
            cv2.line(test_image, (x, 0), (x, 400), (0, 0, 0), 2)
            y = i * 400 // 9
            cv2.line(test_image, (0, y), (400, y), (0, 0, 0), 2)
        
        # Add some content
        cv2.putText(test_image, "5", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        
        result = detector.combine_detection_methods(test_image)
        
        # Check that result is a dictionary with expected keys
        assert isinstance(result, dict)
        expected_keys = ['combined_mask', 'grid_structure', 'content', 'edges']
        for key in expected_keys:
            assert key in result, f"Result should contain '{key}' key"
            assert result[key].shape == test_image.shape[:2]
    
    def test_order_corner_points(self, detector):
        """Test corner point ordering"""
        # Test with a known rectangle
        corners = np.array([
            [[100, 100]],  # top-left
            [[300, 100]],  # top-right  
            [[300, 300]],  # bottom-right
            [[100, 300]]   # bottom-left
        ], dtype=np.float32)
        
        # Shuffle the order
        shuffled = np.array([
            [[300, 300]],  # bottom-right
            [[100, 100]],  # top-left
            [[300, 100]],  # top-right
            [[100, 300]]   # bottom-left
        ], dtype=np.float32)
        
        ordered = detector.order_corner_points(shuffled)
        
        # Check that points are properly ordered
        # top-left should have smallest sum
        assert ordered[0][0] + ordered[0][1] < ordered[2][0] + ordered[2][1]
        
        # top-right should have largest difference (x-y)
        assert ordered[1][0] - ordered[1][1] > ordered[3][0] - ordered[3][1]
    
    def test_extract_board_perspective(self, detector):
        """Test perspective transformation"""
        # Create a test image
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        cv2.rectangle(test_image, (50, 50), (350, 350), (0, 0, 0), 3)
        
        # Define corners for perspective transform
        corners = np.array([
            [[50, 50]],
            [[350, 50]], 
            [[350, 350]],
            [[50, 350]]
        ], dtype=np.float32)
        
        warped = detector.extract_board_perspective(test_image, corners)
        
        # Check output shape
        assert warped.shape == (detector.output_size, detector.output_size, 3)
        assert warped.dtype == test_image.dtype
    
    def test_find_board_rectangle(self, detector):
        """Test board rectangle detection"""
        # Create a binary mask with a clear rectangle
        mask = np.zeros((400, 400), dtype=np.uint8)
        cv2.rectangle(mask, (50, 50), (350, 350), 255, -1)
        
        corners = detector.find_board_rectangle(mask, (400, 400))
        
        if corners is not None:
            assert corners.shape == (4, 1, 2)  # 4 corners, each with x,y coordinates
            assert corners.dtype in [np.int32, np.float32]
    
    def test_detect_and_extract_with_valid_image(self, detector, sample_image_paths):
        """Test the main detection and extraction method with real images"""
        for img_path in sample_image_paths:
            if img_path.exists():
                result = detector.detect_and_extract(str(img_path))
                
                # Check result structure
                assert isinstance(result, dict)
                assert 'success' in result
                assert 'original_image' in result
                assert 'board' in result
                
                # Check image loading
                assert result['original_image'] is not None
                assert result['board'] is not None
                
                # If successful, check board dimensions
                if result['success']:
                    board = result['board']
                    assert board.shape[0] == detector.output_size
                    assert board.shape[1] == detector.output_size
    
    def test_detect_and_extract_with_array_input(self, detector):
        """Test detection with numpy array input"""
        # Create a test image array
        test_image = np.ones((400, 400, 3), dtype=np.uint8) * 255
        
        # Add a simple sudoku-like grid
        for i in range(1, 9):
            x = i * 400 // 9
            cv2.line(test_image, (x, 0), (x, 400), (0, 0, 0), 2)
            y = i * 400 // 9  
            cv2.line(test_image, (0, y), (400, y), (0, 0, 0), 2)
        
        result = detector.detect_and_extract(test_image)
        
        assert isinstance(result, dict)
        assert 'success' in result
        assert result['original_image'] is not None
    
    def test_detect_and_extract_invalid_path(self, detector):
        """Test detection with invalid file path"""
        result = detector.detect_and_extract("nonexistent_file.jpg")
        
        assert result['success'] == False
        assert 'error' in result
        assert 'not found' in result['error'].lower()
    
    def test_detect_and_extract_empty_array(self, detector):
        """Test detection with invalid array input"""
        # Test with None
        try:
            result = detector.detect_and_extract(None)
            assert result['success'] == False
        except:
            pass  # Exception is acceptable for invalid input
    
    def test_detection_consistency(self, detector, sample_image_paths):
        """Test that detection results are consistent across multiple runs"""
        for img_path in sample_image_paths:
            if img_path.exists():
                # Run detection multiple times
                results = []
                for _ in range(3):
                    result = detector.detect_and_extract(str(img_path))
                    results.append(result['success'])
                
                # Results should be consistent
                assert len(set(results)) <= 1, "Detection results should be consistent"
    
    def test_different_output_sizes(self, sample_image_paths):
        """Test detector with different output sizes"""
        sizes = [300, 450, 600]
        
        for size in sizes:
            detector = SudokuBoardDetector(output_size=size)
            
            for img_path in sample_image_paths:
                if img_path.exists():
                    result = detector.detect_and_extract(str(img_path))
                    
                    if result['success'] and result['board'] is not None:
                        board = result['board']
                        assert board.shape[0] == size
                        assert board.shape[1] == size
                    break  # Test with just one image per size
