# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Main Application Package for Sudoku Detection and Solving System.

This package contains the complete Sudoku Detection and Solving application,
integrating computer vision, OCR, mathematical solving algorithms, and a 
web-based user interface. The system provides end-to-end functionality from
image upload to solution presentation.

Package Structure:
    gui/: Streamlit-based web interface components
    src/: Core processing modules (detection, scanning, solving)
    utils/: Utility functions and validation tools
    tests/: Comprehensive test suite for all modules
    official_models/: Pre-trained OCR and detection models

Key Features:
- Image-based Sudoku puzzle detection and extraction
- OCR-powered digit recognition from puzzle images
- Advanced backtracking and constraint-based solving algorithms
- Interactive web interface with real-time validation
- Comprehensive testing and validation framework
- Modular architecture for easy maintenance and extension

Workflow:
1. Image upload and preprocessing
2. Grid detection and perspective correction
3. Cell extraction and digit recognition via OCR
4. Interactive puzzle editing and validation
5. Automated solving with multiple algorithms
6. Solution presentation and export

Author: Tomáš Motus
Project: Sudoku Detection and Solving System
Version: 1.0
"""
