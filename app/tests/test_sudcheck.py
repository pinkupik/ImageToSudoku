# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Test Module for Sudoku Validation and Checking Utilities.

This module contains comprehensive test cases for the Sudoku validation system,
which provides constraint checking and board state verification functionality.
It validates the mathematical rules and logic used throughout the application.

Test Coverage:
- Individual constraint validation (rows, columns, subgrids)
- Complete board validation logic
- Empty board detection
- Solution completeness verification
- Edge cases and invalid inputs
- Performance testing with various board configurations

The tests ensure the reliability and accuracy of the validation system across
different board states, from empty puzzles to complete solutions, including
various invalid configurations to verify proper error detection.

Dependencies:
- pytest for test framework
- numpy for array operations
- sudcheck utilities for validation functions

Author: Tomáš Motus
Project: Sudoku Detection and Solving System
"""
import sys
import os
import pytest
import numpy as np

# Add the parent directory to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import sudcheck as scheck

# @generated "[partially]" Copilot Claude Sonnet 4: Add more tests
class TestSudokuValidation:
    """Test suite for Sudoku validation functions"""
    
    @pytest.fixture
    def valid_partial_board(self):
        """A valid partial Sudoku board"""
        return np.array([
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
    
    @pytest.fixture
    def solved_board(self):
        """A completely solved valid Sudoku board"""
        return np.array([
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ])
    
    @pytest.fixture
    def empty_board(self):
        """A completely empty Sudoku board"""
        return np.zeros((9, 9), dtype=int)
    
    @pytest.fixture
    def invalid_row_board(self):
        """A board with duplicate in a row"""
        return np.array([
            [5, 5, 0, 0, 0, 0, 0, 0, 0],  # Two 5s in first row
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    
    @pytest.fixture
    def invalid_column_board(self):
        """A board with duplicate in a column"""
        return np.array([
            [5, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 0, 0],  # Two 5s in first column
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    
    @pytest.fixture
    def invalid_subgrid_board(self):
        """A board with duplicate in a 3x3 subgrid"""
        return np.array([
            [5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0],  # Two 5s in top-left subgrid
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])

    def test_is_valid_sudoku_with_valid_partial_board(self, valid_partial_board):
        """Test is_valid_sudoku with a valid partial board"""
        assert scheck.is_valid_sudoku(valid_partial_board) == True
    
    def test_is_valid_sudoku_with_solved_board(self, solved_board):
        """Test is_valid_sudoku with a completely solved board"""
        assert scheck.is_valid_sudoku(solved_board) == True
    
    def test_is_valid_sudoku_with_empty_board(self, empty_board):
        """Test is_valid_sudoku with an empty board"""
        assert scheck.is_valid_sudoku(empty_board) == True
    
    def test_is_valid_sudoku_with_invalid_row(self, invalid_row_board):
        """Test is_valid_sudoku with duplicate in row"""
        assert scheck.is_valid_sudoku(invalid_row_board) == False
    
    def test_is_valid_sudoku_with_invalid_column(self, invalid_column_board):
        """Test is_valid_sudoku with duplicate in column"""
        assert scheck.is_valid_sudoku(invalid_column_board) == False
    
    def test_is_valid_sudoku_with_invalid_subgrid(self, invalid_subgrid_board):
        """Test is_valid_sudoku with duplicate in subgrid"""
        assert scheck.is_valid_sudoku(invalid_subgrid_board) == False
    
    def test_is_valid_sudoku_with_invalid_values(self):
        """Test is_valid_sudoku with out-of-range values"""
        invalid_board = np.array([
            [10, 0, 0, 0, 0, 0, 0, 0, 0],  # 10 is invalid
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        assert scheck.is_valid_sudoku(invalid_board) == False
    
    def test_is_valid_sudoku_with_negative_values(self):
        """Test is_valid_sudoku with negative values"""
        invalid_board = np.array([
            [-1, 0, 0, 0, 0, 0, 0, 0, 0],  # -1 is invalid
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        assert scheck.is_valid_sudoku(invalid_board) == False

    def test_is_valid_group_with_valid_group(self):
        """Test is_valid_group with a valid group"""
        valid_group = np.array([1, 2, 3, 0, 0, 0, 0, 0, 0])
        assert scheck.is_valid_group(valid_group) == True
    
    def test_is_valid_group_with_complete_group(self):
        """Test is_valid_group with a complete valid group"""
        complete_group = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        assert scheck.is_valid_group(complete_group) == True
    
    def test_is_valid_group_with_duplicate(self):
        """Test is_valid_group with duplicate values"""
        invalid_group = np.array([1, 1, 3, 0, 0, 0, 0, 0, 0])
        assert scheck.is_valid_group(invalid_group) == False
    
    def test_is_valid_group_with_invalid_value(self):
        """Test is_valid_group with out-of-range value"""
        invalid_group = np.array([1, 10, 3, 0, 0, 0, 0, 0, 0])
        assert scheck.is_valid_group(invalid_group) == False
    
    def test_is_valid_group_with_all_zeros(self):
        """Test is_valid_group with all empty cells"""
        empty_group = np.zeros(9, dtype=int)
        assert scheck.is_valid_group(empty_group) == True
    
    def test_is_valid_group_with_negative_value(self):
        """Test is_valid_group with negative value"""
        invalid_group = np.array([-1, 2, 3, 0, 0, 0, 0, 0, 0])
        assert scheck.is_valid_group(invalid_group) == False

    def test_is_nonempty_sudoku_with_partial_board(self, valid_partial_board):
        """Test is_nonempty_sudoku with a partial board"""
        assert scheck.is_nonempty_sudoku(valid_partial_board) == True
    
    def test_is_nonempty_sudoku_with_solved_board(self, solved_board):
        """Test is_nonempty_sudoku with a solved board"""
        assert scheck.is_nonempty_sudoku(solved_board) == True
    
    def test_is_nonempty_sudoku_with_empty_board(self, empty_board):
        """Test is_nonempty_sudoku with an empty board"""
        assert scheck.is_nonempty_sudoku(empty_board) == False
    
    def test_is_nonempty_sudoku_with_single_value(self):
        """Test is_nonempty_sudoku with only one filled cell"""
        single_value_board = np.zeros((9, 9), dtype=int)
        single_value_board[0, 0] = 5
        assert scheck.is_nonempty_sudoku(single_value_board) == True
    
    def test_is_nonempty_sudoku_with_zero_size_array(self):
        """Test is_nonempty_sudoku with zero-size array"""
        empty_array = np.array([])
        assert scheck.is_nonempty_sudoku(empty_array) == False

    def test_is_solved_sudoku_with_solved_board(self, solved_board):
        """Test is_solved_sudoku with a complete valid solution"""
        assert scheck.is_solved_sudoku(solved_board) == True
    
    def test_is_solved_sudoku_with_partial_board(self, valid_partial_board):
        """Test is_solved_sudoku with a partial board"""
        assert scheck.is_solved_sudoku(valid_partial_board) == False
    
    def test_is_solved_sudoku_with_empty_board(self, empty_board):
        """Test is_solved_sudoku with an empty board"""
        assert scheck.is_solved_sudoku(empty_board) == False
    
    def test_is_solved_sudoku_with_complete_invalid_board(self):
        """Test is_solved_sudoku with a complete but invalid board"""
        invalid_solved_board = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # All 1s - invalid
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            [5, 5, 5, 5, 5, 5, 5, 5, 5],
            [6, 6, 6, 6, 6, 6, 6, 6, 6],
            [7, 7, 7, 7, 7, 7, 7, 7, 7],
            [8, 8, 8, 8, 8, 8, 8, 8, 8],
            [9, 9, 9, 9, 9, 9, 9, 9, 9]
        ])
        assert scheck.is_solved_sudoku(invalid_solved_board) == False
    
    def test_is_solved_sudoku_with_almost_complete_board(self):
        """Test is_solved_sudoku with one missing cell"""
        almost_complete = np.array([
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 0]  # Last cell is empty
        ])
        assert scheck.is_solved_sudoku(almost_complete) == False

    def test_all_subgrids_validation(self, solved_board):
        """Test that all 9 subgrids are properly validated"""
        # Test each subgrid individually
        for i in range(3):
            for j in range(3):
                subgrid = solved_board[i*3:(i+1)*3, j*3:(j+1)*3].flatten()
                assert scheck.is_valid_group(subgrid) == True
    
    def test_subgrid_validation_with_duplicates(self):
        """Test subgrid validation with duplicates in different subgrids"""
        board_with_subgrid_duplicate = np.array([
            [5, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 5, 0, 0, 0, 0, 0, 0],  # 5 appears twice in top-left subgrid
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
        assert scheck.is_valid_sudoku(board_with_subgrid_duplicate) == False
    
    def test_validation_with_different_dtypes(self):
        """Test validation with different numpy data types"""
        # Test with float array
        float_board = np.array([
            [5.0, 3.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 0.0],
            [6.0, 0.0, 0.0, 1.0, 9.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 9.0, 8.0, 0.0, 0.0, 0.0, 0.0, 6.0, 0.0],
            [8.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 3.0],
            [4.0, 0.0, 0.0, 8.0, 0.0, 3.0, 0.0, 0.0, 1.0],
            [7.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 6.0],
            [0.0, 6.0, 0.0, 0.0, 0.0, 0.0, 2.0, 8.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 1.0, 9.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, 0.0, 8.0, 0.0, 0.0, 7.0, 9.0]
        ])
        assert scheck.is_valid_sudoku(float_board) == True
    
    def test_performance_with_large_number_of_validations(self):
        """Test performance with repeated validations"""
        import time
        
        # Create several different boards
        boards = [
            np.zeros((9, 9), dtype=int),
            np.ones((9, 9), dtype=int),
            np.random.randint(0, 10, (9, 9))
        ]
        
        start_time = time.time()
        
        # Perform many validations
        for _ in range(100):
            for board in boards:
                scheck.is_valid_sudoku(board)
                scheck.is_nonempty_sudoku(board)
                scheck.is_solved_sudoku(board)
        
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 5.0, "Validation performance too slow"
    
    def test_consistency_across_multiple_calls(self, valid_partial_board, solved_board, empty_board):
        """Test that validation functions are consistent across multiple calls"""
        # Test multiple calls return the same result
        for _ in range(10):
            assert scheck.is_valid_sudoku(valid_partial_board) == True
            assert scheck.is_valid_sudoku(solved_board) == True
            assert scheck.is_valid_sudoku(empty_board) == True
            
            assert scheck.is_nonempty_sudoku(valid_partial_board) == True
            assert scheck.is_nonempty_sudoku(solved_board) == True
            assert scheck.is_nonempty_sudoku(empty_board) == False
            
            assert scheck.is_solved_sudoku(valid_partial_board) == False
            assert scheck.is_solved_sudoku(solved_board) == True
            assert scheck.is_solved_sudoku(empty_board) == False
    
    def test_edge_case_values(self):
        """Test validation with edge case values"""
        # Test board with all 9s (valid if no duplicates)
        all_nines_board = np.zeros((9, 9), dtype=int)
        all_nines_board[0, :] = 9  # One row of 9s
        assert scheck.is_valid_sudoku(all_nines_board) == False
        
        # Test board with mix of valid edge values
        edge_board = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0, 9],  # Min and max values
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [9, 0, 0, 0, 0, 0, 0, 0, 1]
        ])
        assert scheck.is_valid_sudoku(edge_board) == True

def test_integration_with_solving_validation():
    """Integration test to verify validation works with solved puzzles"""
    # This would normally import a solver, but we'll create a known solution
    solved_puzzle = np.array([
        [5, 3, 4, 6, 7, 8, 9, 1, 2],
        [6, 7, 2, 1, 9, 5, 3, 4, 8],
        [1, 9, 8, 3, 4, 2, 5, 6, 7],
        [8, 5, 9, 7, 6, 1, 4, 2, 3],
        [4, 2, 6, 8, 5, 3, 7, 9, 1],
        [7, 1, 3, 9, 2, 4, 8, 5, 6],
        [9, 6, 1, 5, 3, 7, 2, 8, 4],
        [2, 8, 7, 4, 1, 9, 6, 3, 5],
        [3, 4, 5, 2, 8, 6, 1, 7, 9]
    ])
    
    # All validation functions should work correctly
    assert scheck.is_valid_sudoku(solved_puzzle) == True
    assert scheck.is_nonempty_sudoku(solved_puzzle) == True
    assert scheck.is_solved_sudoku(solved_puzzle) == True

@pytest.mark.parametrize("board_type,expected_valid,expected_nonempty,expected_solved", [
    ("empty", True, False, False),
    ("partial", True, True, False),
    ("solved", True, True, True),
    ("invalid_row", False, True, False),
    ("invalid_column", False, True, False),
    ("invalid_subgrid", False, True, False),
])
def test_validation_combinations(board_type, expected_valid, expected_nonempty, expected_solved):
    """Test various board types with expected validation results"""
    board = None
    if board_type == "empty":
        board = np.zeros((9, 9), dtype=int)
    elif board_type == "partial":
        board = np.array([
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
    elif board_type == "solved":
        board = np.array([
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ])
    elif board_type == "invalid_row":
        board = np.array([
            [5, 5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    elif board_type == "invalid_column":
        board = np.array([
            [5, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    elif board_type == "invalid_subgrid":
        board = np.array([
            [5, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 5, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0]
        ])
    
    assert scheck.is_valid_sudoku(board) == expected_valid
    assert scheck.is_nonempty_sudoku(board) == expected_nonempty
    assert scheck.is_solved_sudoku(board) == expected_solved
