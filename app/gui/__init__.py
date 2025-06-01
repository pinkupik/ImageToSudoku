# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
GUI Package for Sudoku Detection and Solving Application.

This package contains all Streamlit-based graphical user interface components
for the Sudoku Detection and Solving system. It provides a complete web-based
interface that allows users to upload images, detect Sudoku puzzles, manually
input or edit puzzles, and view solved results.

Modules:
    display: Main display controller and image processing interface
    dtables: Two-column layout manager for puzzle display and interaction
    dinput: Interactive 9x9 grid for manual puzzle input and editing
    dsolved: Read-only display for solved Sudoku puzzles
    dempty: Empty grid placeholder display

Key Features:
- Streamlit web-based interface
- Image upload and processing workflow
- Interactive puzzle input and validation
- Real-time solving and result display
- Responsive layout design
- Session state management
- Error handling and user feedback

Architecture:
The GUI follows a modular design where each component handles specific
aspects of the user interface. The display module serves as the main
controller, coordinating between other modules to provide a seamless
user experience from image upload to solution presentation.

Author: Tomás Motus
Project: Sudoku Detection and Solving System
"""
