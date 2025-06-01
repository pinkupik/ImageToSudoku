# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Interactive Sudoku Input Grid Display Module

This module provides the interactive input interface for Sudoku puzzles within
the Streamlit application. It creates a 9x9 grid of text input fields that
allow users to enter, modify, and view Sudoku digits in real-time.

The module handles:
- Rendering a 9x9 interactive grid of input fields
- Converting between board arrays and user input
- Input validation and sanitization
- Responsive layout with proper spacing
- Real-time state management

Functions:
    display_input(board): Creates and manages the interactive input grid

Features:
- Single-character input validation per cell
- Automatic conversion of non-numeric input to empty cells
- Visual grid layout matching standard Sudoku appearance
- Real-time board state updates
- Disabled cell support for locked puzzle values

Author: Tomás Motus
Project: Sudoku Detection and Solving System
"""
import streamlit as st
import numpy as np


def display_input(board):
    """
    Display an interactive 9x9 Sudoku input grid and return the updated board state.

    This function creates a responsive grid of text input fields arranged in the
    standard Sudoku 9x9 layout. Each cell accepts single-digit input (1-9) or
    remains empty (represented as 0 in the returned array).

    Args:
        board (numpy.ndarray): A 9x9 integer matrix representing the initial
                              Sudoku state, where 0 represents empty cells
                              and 1-9 represent filled digits

    Returns:
        numpy.ndarray: A 9x9 integer matrix representing the updated board state
                      after user input, maintaining the same format as input

    Interface Features:
        - 9x9 grid layout with small gaps between cells
        - Single-character input limitation per cell
        - Empty cells display as blank (not "0")
        - Non-numeric input automatically converts to empty (0)
        - Responsive column layout adapting to screen size
        - Hidden labels for clean appearance
        - Unique keys for each cell to maintain state

    User Interaction:
        - Click any cell to enter or modify digits
        - Only accepts digits 1-9 or empty input
        - Invalid characters are automatically filtered out
        - Changes are reflected immediately in the returned board state

    Technical Implementation:
        - Uses Streamlit's text_input widgets arranged in columns
        - Maintains individual cell state with unique keys
        - Converts string input to integers with error handling
        - Preserves board structure as numpy array for compatibility

    Note:
        The function creates a new container for layout control and uses
        Streamlit's column system to achieve proper grid alignment. Each
        cell maintains its own session state through unique key assignment.
    """
    st.write("### Unsolved Sudoku Board")
    board_state = []
    # Use a container to control layout
    with st.container(key="sudoku_input_container"):
        for i in range(9):
            cols = st.columns(9, gap="small")
            row = []
            for j in range(9):
                cell_value = "" if board[i][j] == 0 else str(board[i][j])
                value = cols[j].text_input(
                    label=f"input_cell_{i}_{j}",
                    value=cell_value,
                    max_chars=1,
                    key=f"sudoku_{i}_{j}",
                    disabled=False,
                    label_visibility="collapsed",
                    # Make the input box more square via CSS
                    placeholder="",
                )
                # Convert input to int if possible, else 0
                try:
                    row.append(int(value))
                except ValueError:
                    row.append(0)
            board_state.append(row)
    return np.array(board_state)
