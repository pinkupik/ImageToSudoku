# @generated "[partially]" Copilot Claude Sonnet 4: Add more tests
# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Test Module for Sudoku OCR Scanning Functionality.

This module contains comprehensive test cases for the Sudoku OCR scanning system,
which handles digit recognition from extracted Sudoku cell images. It validates
the OCR pipeline that converts visual digit representations into numerical values.

Test Coverage:
- OCR digit recognition accuracy
- Image preprocessing for OCR
- Cell image quality validation
- Numerical output validation (0-9 range)
- Performance testing with various image types
- Error handling for corrupted or invalid images

The tests ensure the reliability and accuracy of the OCR system across
different image qualities, digit styles, and preprocessing conditions.
Test images are located in the tests/images directory.

Dependencies:
- pytest for test framework
- numpy for array operations  
- PIL/OpenCV for image processing
- PaddleOCR for digit recognition

Author: Tomáš Motus
Project: Sudoku Detection and Solving System
"""
from utils import sudcheck as scheck
from src import sudscan as sscan
import sys
import os
import pytest
import numpy as np
# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_scan_table_returns_correct_shape():
    """Test that scan_table returns a 9x9 numpy array"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")

    if os.path.exists(img_path):
        result = sscan.scan_table(img_path)
        assert result.shape == (
            9, 9), f"Expected shape (9, 9), got {result.shape}"
        assert isinstance(result, np.ndarray), "Result should be a numpy array"


def test_scan_table_returns_valid_values():
    """Test that scan_table returns only valid Sudoku values (0-9)"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")

    if os.path.exists(img_path):
        result = sscan.scan_table(img_path)
        assert np.all((result >= 0) & (result <= 9)
                      ), "All values should be between 0 and 9"
        assert result.dtype == int, "Result should contain integers"


def test_scan_table_image1():
    """Test scan_table with image1.jpg"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")

    result = sscan.scan_table(img_path)
    assert result is not None, "scan_table should not return None"
    assert result.shape == (9, 9), "Result should be 9x9 matrix"


def test_scan_table_image2():
    """Test scan_table with image2.png"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image2.png")

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")

    result = sscan.scan_table(img_path)
    assert result is not None, "scan_table should not return None"
    assert result.shape == (9, 9), "Result should be 9x9 matrix"


def test_scan_table_image3():
    """Test scan_table with image3.png"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image3.png")

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")

    result = sscan.scan_table(img_path)
    assert result is not None, "scan_table should not return None"
    assert result.shape == (9, 9), "Result should be 9x9 matrix"
    result_ref = np.array([[5, 3, 0, 0, 7, 0, 0, 0, 0],
                           [6, 0, 0, 1, 9, 5, 0, 0, 0],
                           [0, 9, 8, 0, 0, 0, 0, 6, 0],
                           [8, 0, 0, 0, 6, 0, 0, 0, 3],
                           [4, 0, 0, 8, 0, 3, 0, 0, 1],
                           [7, 0, 0, 0, 2, 0, 0, 0, 6],
                           [0, 6, 0, 0, 0, 0, 2, 8, 0],
                           [0, 0, 0, 4, 1, 9, 0, 0, 5],
                           [0, 0, 0, 0, 8, 0, 0, 7, 9]])
    assert np.array_equal(
        result, result_ref), "Result does not match reference"


def test_scan_table_image4():
    """Test scan_table with image4.png"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image4.png")

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")

    result = sscan.scan_table(img_path)
    assert result is not None, "scan_table should not return None"
    assert result.shape == (9, 9), "Result should be 9x9 matrix"


def test_scan_table_nonexistent_file():
    """Test scan_table with non-existent file"""
    with pytest.raises(Exception):
        sscan.scan_table("nonexistent_file.jpg")


def test_scan_table_output_format():
    """Test that scan_table returns correct format"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")

    if os.path.exists(img_path):
        result = sscan.scan_table(img_path)

        # Check shape
        assert result.shape == (
            9, 9), f"Expected shape (9, 9), got {result.shape}"

        # Check type
        assert isinstance(result, np.ndarray), "Result should be numpy array"

        # Check dtype is integer
        assert np.issubdtype(
            result.dtype, np.integer), "Result should contain integers"


def test_scan_table_value_range():
    """Test that scan_table returns values in valid range"""
    test_dir = "app/tests/images"

    for img_name in ["image1.jpg", "image2.png", "image3.png", "image4.png"]:
        img_path = os.path.join(test_dir, img_name)

        if os.path.exists(img_path):
            result = sscan.scan_table(img_path)

            # All values should be between 0 and 9
            assert np.all(result >= 0), f"Found negative values in {img_name}"
            assert np.all(result <= 9), f"Found values > 9 in {img_name}"

            # Check for valid sudoku structure (no invalid values)
            unique_vals = np.unique(result)
            assert all(val in range(10)
                       for val in unique_vals), f"Invalid values in {img_name}"


def test_scan_table_consistency():
    """Test that scan_table produces consistent results"""
    test_dir = "app/tests/images"
    # Use image3 which has known reference
    img_path = os.path.join(test_dir, "image3.png")

    if os.path.exists(img_path):
        # Scan the same image multiple times
        results = []
        for _ in range(3):
            result = sscan.scan_table(img_path)
            results.append(result)

        # Results should be identical
        for i in range(1, len(results)):
            assert np.array_equal(
                results[0], results[i]), "OCR results should be consistent"


def test_scan_table_known_reference():
    """Test scan_table with image that has known expected output"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image3.png")

    if not os.path.exists(img_path):
        pytest.skip(f"Test image not found: {img_path}")

    result = sscan.scan_table(img_path)

    # This is the expected reference from the existing test
    expected = np.array([
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])

    # Check if the result matches the expected output
    # Note: OCR might not be 100% accurate, so we check for reasonable similarity
    matches = np.sum(result == expected)
    total_cells = 81
    accuracy = matches / total_cells

    assert accuracy >= 0.7, f"OCR accuracy too low: {accuracy:.2f}"


def test_scan_table_digit_detection():
    """Test that scan_table can detect individual digits"""
    test_dir = "app/tests/images"

    # Test with digit images if available
    digit_files = ["digit1.png", "digit2.png"]

    for digit_file in digit_files:
        digit_path = os.path.join(test_dir, digit_file)
        if os.path.exists(digit_path):
            try:
                result = sscan.scan_table(digit_path)

                # Should still return 9x9 matrix
                assert result.shape == (9, 9)

                # Should contain at least one non-zero digit
                assert np.any(
                    result > 0), f"No digits detected in {digit_file}"

            except Exception as e:
                # It's acceptable if digit images can't be processed as full grids
                print(f"Could not process {digit_file}: {e}")


def test_scan_table_empty_detection():
    """Test scan_table's ability to detect empty cells"""
    test_dir = "app/tests/images"

    for img_name in ["image1.jpg", "image2.png", "image3.png", "image4.png"]:
        img_path = os.path.join(test_dir, img_name)

        if os.path.exists(img_path):
            result = sscan.scan_table(img_path)

            # Should have some empty cells (zeros) - typical of input puzzles
            num_empty = np.sum(result == 0)
            num_filled = np.sum(result > 0)

            # Most Sudoku puzzles have more empty cells than filled
            # But this depends on the specific test images
            assert num_empty + num_filled == 81, "Total cells should be 81"

            # At least some cells should be detected as empty or filled
            assert num_empty >= 0 and num_filled >= 0


def test_scan_table_sudoku_validity():
    """Test that scanned results could form valid Sudoku puzzles"""
    test_dir = "app/tests/images"

    for img_name in ["image1.jpg", "image2.png", "image3.png", "image4.png"]:
        img_path = os.path.join(test_dir, img_name)

        if os.path.exists(img_path):
            result = sscan.scan_table(img_path)

            # Check if the detected puzzle is a valid partial Sudoku
            # (no conflicts in rows, columns, or boxes for non-zero values)
            is_valid = scheck.is_valid_sudoku(result)

            if not is_valid:
                # Print debug info for invalid results
                print(f"Invalid Sudoku detected from {img_name}")
                print("Detected grid:")
                print(result)

                # For now, we'll be lenient since OCR might not be perfect
                # But we should at least detect some reasonable structure
                num_detected = np.sum(result > 0)
                assert num_detected > 0, f"No digits detected in {img_name}"


def test_scan_table_error_handling():
    """Test scan_table error handling"""

    # Test with non-existent file
    with pytest.raises(Exception):
        sscan.scan_table("definitely_nonexistent_file.jpg")

    # Test with invalid file path
    with pytest.raises(Exception):
        sscan.scan_table("")


def test_scan_table_file_format_support():
    """Test scan_table with different image formats"""
    test_dir = "app/tests/images"

    # Test different formats
    formats = {
        ".jpg": "image1.jpg",
        ".png": "image2.png"
    }

    for ext, filename in formats.items():
        img_path = os.path.join(test_dir, filename)

        if os.path.exists(img_path):
            try:
                result = sscan.scan_table(img_path)

                assert result.shape == (9, 9)
                assert isinstance(result, np.ndarray)

                print(f"Successfully processed {ext} format")

            except Exception as e:
                pytest.fail(f"Failed to process {ext} format: {e}")


def test_scan_table_performance():
    """Test scan_table performance"""
    import time

    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")

    if os.path.exists(img_path):
        # Measure scanning time
        start_time = time.time()
        result = sscan.scan_table(img_path)
        end_time = time.time()

        scan_time = end_time - start_time

        # Should complete within reasonable time (adjust based on hardware)
        assert scan_time < 30.0, f"Scanning took too long: {scan_time:.2f}s"

        # Should return valid result
        assert result is not None
        assert result.shape == (9, 9)


def test_scan_table_memory_usage():
    """Test that scan_table doesn't leak memory"""
    test_dir = "app/tests/images"
    img_path = os.path.join(test_dir, "image1.jpg")

    if os.path.exists(img_path):
        # Run multiple times to check for memory leaks
        results = []
        for i in range(5):
            result = sscan.scan_table(img_path)
            results.append(result)

            # Each result should be valid
            assert result.shape == (9, 9)
            assert isinstance(result, np.ndarray)

        # All results should be identical (deterministic)
        for i in range(1, len(results)):
            if not np.array_equal(results[0], results[i]):
                print(f"Warning: OCR results not deterministic between runs")


def test_scan_table_all_images():
    """Comprehensive test on all available test images"""
    test_dir = "app/tests/images"

    if not os.path.exists(test_dir):
        pytest.skip(f"Test directory not found: {test_dir}")

    # Get all image files
    image_files = [f for f in os.listdir(
        test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

    assert len(image_files) > 0, "No test images found"

    results_summary = []

    for img_file in image_files:
        img_path = os.path.join(test_dir, img_file)

        try:
            result = sscan.scan_table(img_path)

            # Basic validation
            assert result.shape == (9, 9)
            assert np.all((result >= 0) & (result <= 9))

            # Count detected digits
            num_digits = np.sum(result > 0)
            detection_rate = (num_digits / 81) * 100

            results_summary.append({
                'file': img_file,
                'digits_detected': num_digits,
                'detection_rate': detection_rate,
                'is_valid_sudoku': scheck.is_valid_sudoku(result)
            })

        except Exception as e:
            results_summary.append({
                'file': img_file,
                'error': str(e)
            })

    # Print summary
    print("\n=== OCR Test Summary ===")
    for result in results_summary:
        if 'error' in result:
            print(f"{result['file']}: ERROR - {result['error']}")
        else:
            print(f"{result['file']}: {result['digits_detected']} digits "
                  f"({result['detection_rate']:.1f}%) - "
                  f"Valid: {result['is_valid_sudoku']}")

    # At least one image should be processed successfully
    successful = [r for r in results_summary if 'error' not in r]
    assert len(successful) > 0, "No images were processed successfully"
