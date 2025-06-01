# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Sudoku Tables Display Management Module

This module orchestrates the display of Sudoku tables in a two-column layout,
managing the interaction between input grids, validation, solving, and result
presentation within the Streamlit interface.

The module coordinates:
- Input table display and user interaction
- Real-time Sudoku validation
- Automatic solving when valid input is provided
- Error handling and user feedback
- Empty state display when no valid input exists

Functions:
    display_tables(sudoku_board): Main function to render the complete table interface

Dependencies:
    - dinput: Handles interactive input grid display
    - dsolved: Manages solved sudoku presentation
    - dempty: Displays empty state interface
    - sudsolve: Provides sudoku solving algorithms
    - sudcheck: Validates sudoku grids and solutions

Author: Tomáš Motus
Project: Sudoku Detection and Solving System
"""
import streamlit as st
from app.gui import dinput, dsolved, dempty
from app.src import sudsolve as ssolve
from app.utils import sudcheck


def display_tables(sudoku_board):
    """
    Display and manage the complete Sudoku table interface in a two-column layout.

    This function creates a responsive two-column interface that handles the entire
    Sudoku solving workflow:

    Left Column (Input):
    - Displays an interactive input grid based on the provided sudoku_board
    - Allows users to modify cell values
    - Returns updated board state for processing

    Right Column (Output):
    - Validates the input board for completeness and correctness
    - Automatically solves valid puzzles
    - Displays solved solutions or appropriate error messages
    - Shows empty state when no valid input is available

    Args:
        sudoku_board (numpy.ndarray): A 9x9 integer matrix representing the initial
                                    Sudoku state, where 0 represents empty cells
                                    and 1-9 represent filled digits

    Workflow:
        1. Creates two-column layout with large gap
        2. Left column: Renders interactive input grid, captures user modifications
        3. Right column: Validates input and determines appropriate display:
           - Non-empty + Valid → Attempts to solve and shows solution
           - Non-empty + Invalid → Shows validation error
           - Empty/Incomplete → Shows empty state placeholder
           - Unsolvable → Shows unsolvable error message

    User Experience:
        - Real-time validation feedback
        - Immediate solving when conditions are met
        - Clear error messages for invalid inputs
        - Responsive layout adapting to screen size

    Note:
        The function relies on external validation (sudcheck) to determine
        board states and uses the sudsolve module for actual puzzle solving.
        All user interface rendering is delegated to specialized display modules.
    """
    cols = st.columns(2, gap="large")
    with cols[0]:
        updated_board = dinput.display_input(sudoku_board)
    with cols[1]:
        if sudcheck.is_nonempty_sudoku(updated_board):
            if sudcheck.is_valid_sudoku(updated_board):
                solved_board = ssolve.solve_advanced(updated_board)
                if sudcheck.is_solved_sudoku(solved_board):
                    dsolved.display_solved(solved_board)
                else:
                    st.error("Sudoku is unsolvable. Maybe check your input?")
            else:
                st.error("Invalid Sudoku input! Please check your entries.")
        else:
            dempty.display_empty()
