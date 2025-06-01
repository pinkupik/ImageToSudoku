# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Empty Sudoku Grid Display Module for Streamlit GUI.

This module provides functionality to display an empty 9x9 Sudoku grid
using Streamlit components. The display creates a blank, read-only grid
that can be used as a placeholder or initial state before puzzle detection
and solving operations.

Key Features:
- Empty 9x9 grid display with disabled input fields
- Consistent visual styling with other grid components
- Streamlit container-based layout management
- Read-only presentation for display purposes
- Uniform cell formatting and spacing

The module serves as a visual placeholder in the application workflow,
typically shown when no puzzle has been detected or processed yet.

Author: Tomás Motus
Project: Sudoku Detection and Solving System
"""
import streamlit as st


def display_empty():
    """
    Display an empty Sudoku board in a read-only 9x9 grid format.

    Creates a visual representation of a blank Sudoku puzzle using
    Streamlit's text_input components configured as disabled fields.
    All cells are empty and non-interactive, serving as a placeholder
    or initial state display.

    Args:
        None

    Returns:
        None: Function creates Streamlit UI components directly.

    Note:
        - All input fields are disabled to prevent user interaction
        - All cells are initialized as empty strings
        - Each cell has a unique key for Streamlit state management
        - Uses consistent styling with other grid interfaces
        - Maintains the same visual structure as input and solved grids
    """
    st.write("### Solved Sudoku Board")
    with st.container():
        for i in range(9):
            cols = st.columns(9, gap="small")
            for j in range(9):
                cols[j].text_input(
                    label=f"empty_cell_{i}_{j}",
                    value="",
                    max_chars=1,
                    key=f"sudoku_empty_{i}_{j}",
                    disabled=True,
                    label_visibility="collapsed",
                    placeholder="",
                )
