# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Main entry point for the Sudoku Detection and Solving Application.

This module provides the main entry point for a Streamlit-based web application
that allows users to detect, extract, and solve Sudoku puzzles from images.
The application integrates computer vision, OCR, and solving algorithms to
provide a complete Sudoku processing pipeline.

Usage:
    Run the application using Streamlit:
    $ streamlit run main.py
    
    Or execute directly:
    $ python main.py

Author: Tomas
Project: Sudoku Detection and Solving System
"""
from app.gui import display


def main():
    """
    Main function that initializes and runs the Streamlit GUI application.

    This function serves as the entry point for the Sudoku detection and solving
    application. It launches the Streamlit interface which provides users with
    tools to upload images, detect Sudoku boards, extract grids, and solve puzzles.

    The function delegates the actual GUI rendering to the display module,
    maintaining separation of concerns between the main entry point and
    the user interface logic.
    """
    display.display()


if __name__ == "__main__":
    main()
