# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Solved Sudoku Display Module for Streamlit GUI.

This module provides functionality to display a solved Sudoku puzzle in a 
read-only 9x9 grid format using Streamlit components. The display creates
an immutable grid showing the complete solution with proper visual formatting
and cell organization.

Key Features:
- Read-only 9x9 grid display
- Disabled input fields for solution presentation
- Consistent visual styling with input grid
- Streamlit container-based layout management
- Automatic cell value formatting and validation

The module is designed to present the final solved state of a Sudoku puzzle
after processing through the detection and solving pipeline.

Author: Tomás Motus
Project: Sudoku Detection and Solving System
"""
import streamlit as st


def display_solved(board):
    """
    Display a solved Sudoku board in a read-only 9x9 grid format.

    Creates a visual representation of the solved Sudoku puzzle using
    Streamlit's text_input components configured as disabled fields.
    The display maintains the same visual structure as the input grid
    but prevents user interaction.

    Args:
        board (list[list[int]]): A 9x9 2D list representing the solved 
            Sudoku board where each cell contains integers 1-9, or 0 
            for empty cells (though solved boards should have no zeros).

    Returns:
        None: Function creates Streamlit UI components directly.

    Note:
        - All input fields are disabled to prevent modification
        - Empty cells (value 0) are displayed as blank
        - Each cell has a unique key for Streamlit state management
        - Uses consistent styling with the input grid interface
    """
    st.write("### Solved Sudoku Board")
    with st.container():
        for i in range(9):
            cols = st.columns(9, gap="small")
            for j in range(9):
                cell_value = "" if board[i][j] == 0 else str(board[i][j])
                cols[j].text_input(
                    label=f"solved_cell_{i}_{j}",
                    value=cell_value,
                    max_chars=1,
                    key=f"sudoku_replica_{i}_{j}",
                    disabled=True,
                    label_visibility="collapsed",
                    placeholder="",
                )
