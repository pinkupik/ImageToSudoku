# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Sudoku Validation and Checking Utilities.

This module provides comprehensive validation functions for Sudoku puzzles,
including checking board validity, solution completeness, and constraint
compliance. It implements the core Sudoku rules and validation logic used
throughout the application.

Key Features:
- Complete Sudoku board validation
- Row, column, and 3x3 subgrid constraint checking
- Empty board detection
- Solution completeness verification
- NumPy-based efficient array operations

Validation Rules:
- Each row must contain unique digits 1-9 (excluding zeros)
- Each column must contain unique digits 1-9 (excluding zeros)
- Each 3x3 subgrid must contain unique digits 1-9 (excluding zeros)
- Valid values are integers 1-9, with 0 representing empty cells
- Solved puzzles must have no empty cells and satisfy all constraints

Author: Tomas
Project: Sudoku Detection and Solving System
"""
import numpy as np


def is_valid_sudoku(board):
    """
    Check if a given Sudoku board is valid according to Sudoku constraints.

    Validates that the current state of the Sudoku board follows all rules:
    no duplicate numbers in rows, columns, or 3x3 subgrids. Empty cells 
    (represented by 0) are ignored during validation.

    Args:
        board (np.ndarray): 9x9 Sudoku board with values 0-9, where 0 
            represents empty cells and 1-9 are filled digits.

    Returns:
        bool: True if the board satisfies all Sudoku constraints, 
            False if any constraint is violated.

    Note:
        This function checks partial boards and does not require the
        puzzle to be completely solved. It only verifies that existing
        filled cells follow Sudoku rules.
    """
    # Check rows
    for row in board:
        if not is_valid_group(row):
            return False

    # Check columns
    for col in board.T:
        if not is_valid_group(col):
            return False

    # Check 3x3 subgrids
    for i in range(3):
        for j in range(3):
            subgrid = board[i*3:(i+1)*3, j*3:(j+1)*3].flatten()
            if not is_valid_group(subgrid):
                return False

    return True


def is_valid_group(group):
    """
    Check if a group (row, column, or 3x3 subgrid) satisfies Sudoku constraints.

    Validates that a 1D array representing a Sudoku group contains no duplicate
    values and all values are within the valid range 1-9. Empty cells (0) are
    ignored during validation.

    Args:
        group (np.ndarray): 1D array of 9 Sudoku values, typically representing
            a row, column, or flattened 3x3 subgrid.

    Returns:
        bool: True if the group contains no duplicates and all non-zero values
            are in range 1-9, False otherwise.

    Note:
        - Zero values are treated as empty cells and ignored
        - Only checks for duplicates among filled cells
        - Validates value range for all non-zero entries
    """
    seen = set()
    for value in group:
        if value != 0:  # Ignore empty cells
            if value in seen or not 1 <= value <= 9:
                return False
            seen.add(value)
    return True


def is_nonempty_sudoku(board):
    """
    Check if a Sudoku board contains at least one filled cell.

    Determines whether the board has any non-zero values, which indicates
    that at least some cells have been filled with digits. This is useful
    for validating that a board has been initialized or partially completed.

    Args:
        board (np.ndarray): 9x9 Sudoku board with values 0-9, where 0 
            represents empty cells and 1-9 are filled digits.

    Returns:
        bool: True if at least one cell contains a non-zero value, 
            False if the board is completely empty or has zero size.

    Note:
        - Returns False for empty arrays or arrays with zero size
        - Only checks for presence of non-zero values, not validity
        - Useful for determining if puzzle detection or input has occurred
    """
    # Check if the board is empty
    if board.size == 0 or np.all(board == 0):
        return False
    return True


def is_solved_sudoku(board):
    """
    Check if a Sudoku board is completely solved.

    Determines whether the board represents a valid, complete Sudoku solution
    by verifying that all cells are filled (no zeros) and all Sudoku 
    constraints are satisfied.

    Args:
        board (np.ndarray): 9x9 Sudoku board with values 0-9, where 0 
            represents empty cells and 1-9 are filled digits.

    Returns:
        bool: True if the board is completely filled and satisfies all 
            Sudoku constraints, False otherwise.

    Note:
        - Requires all 81 cells to be filled (no zero values)
        - Must pass all validity checks (rows, columns, subgrids)
        - Combines completeness and correctness verification
        - Used to verify successful puzzle solving
    """
    return np.all(board != 0) and is_valid_sudoku(board)
