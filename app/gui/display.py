# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Main Display Module for Streamlit-based Sudoku Solver GUI

This module provides the primary user interface for the Sudoku Detection and Solving
application. It integrates file upload, camera capture, image processing, Sudoku
detection, and interactive solving capabilities into a comprehensive web interface.

The module handles:
- File upload and camera capture for image input
- Sudoku board detection from images using computer vision
- OCR-based digit extraction from detected boards
- Interactive grid editing and visualization
- Sudoku solving with real-time feedback

Dependencies:
    - Streamlit for web interface
    - OpenCV for image processing
    - NumPy for numerical operations
    - Custom modules for detection, scanning, and table display

Author: Tomáš Motus
Project: Sudoku Detection and Solving System
"""
import os
import streamlit as st
import numpy as np
import cv2
from app.gui import dtables
from app.src import sudscan as sscan
from app.src.suddet import SudokuBoardDetector


def display():
    """
    Main display function that renders the complete Sudoku solver web interface.

    This function creates and manages the entire Streamlit application interface,
    including:
    - Page configuration and layout setup
    - File upload and camera capture functionality
    - Image processing and Sudoku board detection
    - OCR-based digit extraction from images
    - Interactive Sudoku grid display and editing
    - Integration with solving algorithms

    The function maintains session state for camera functionality and handles
    temporary file management for uploaded/captured images. It uses the
    SudokuBoardDetector for computer vision-based board extraction and
    integrates with OCR scanning for digit recognition.

    Interface Features:
    - Wide page layout for optimal viewing
    - File uploader supporting JPG, JPEG, PNG formats
    - Camera capture with toggle functionality
    - Image preview with before/after detection views
    - Extracted Sudoku grid visualization
    - Interactive grid editing capabilities

    Note:
        The function creates temporary files in 'app/utils' directory for
        image processing. These files are managed automatically during
        the detection and scanning process.
    """
    st.set_page_config(layout="wide")  # Use the whole width of the screen
    detector = SudokuBoardDetector(output_size=450, debug=False)
    sudoku_board = np.zeros((9, 9), dtype=int)

    st.header("Crazy AI Sudoku Solver++ Ultra Edition")
    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])
    captured_image = None
    if uploaded_file is None:
        # Show camera input only after pressing "Take a photo"
        if "show_camera" not in st.session_state:
            st.session_state["show_camera"] = False

        if st.button("Take a photo", key="take_photo_button"):
            if not st.session_state["show_camera"]:
                st.session_state["show_camera"] = True
            else:
                st.session_state["show_camera"] = False

        if st.session_state["show_camera"]:
            captured_image = st.camera_input("Or take a photo")
            if captured_image is not None:
                # Hide camera after capture
                st.session_state["show_camera"] = False
        else:
            captured_image = None

    # Prefer captured image if available, else uploaded file
    image_file = captured_image if captured_image is not None else uploaded_file
    if image_file is not None:
        # Save the uploaded file to a temporary location
        temp_image_path = "app/utils"
        with open(os.path.join(temp_image_path, 'temp_uploaded_image.png'), "wb+") as f:
            f.write(image_file.getbuffer())
        cropped_image = detector.detect_and_extract(
            os.path.join(temp_image_path, 'temp_uploaded_image.png'))
        cv2.imwrite(os.path.join(temp_image_path, 'cropped.jpg'),  # pylint:disable=no-member
                    cropped_image['board'])
        with st.expander("Show Uploaded Image"):
            image_cols = st.columns(2, gap="large")
            image_cols[0].image(
                image_file, caption="Uploaded Image", width=300)
            image_cols[1].image(cropped_image['board'],
                                caption="Detected Sudoku Board", width=300)
        # Process the image to extract the Sudoku grid
        sudoku_board = sscan.scan_table(
            os.path.join(temp_image_path, 'cropped.jpg'))
        # Optionally, remove the temporary file after processing
    else:
        # Example empty Sudoku board
        sudoku_board = np.zeros((9, 9), dtype=int)
    # Custom CSS for wider input boxes
    st.markdown("""
        <style>
        div[data-baseweb="input"] input {
            width: 2.5em !important;
            height: 1.18em !important;
            text-align: center !important;
            font-size: 2em !important;
            padding: 0 !important;
        }
        </style>
    """, unsafe_allow_html=True)

    dtables.display_tables(sudoku_board)


if __name__ == "__main__":
    display()
