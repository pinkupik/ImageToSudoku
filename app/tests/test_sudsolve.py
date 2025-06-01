# @generated "[partially]" Copilot Claude Sonnet 4: Add more tests
# @generated "[partially]" Copilot Claude Sonnet 4: Add docstings
"""
Tests for the Advanced Sudoku Solver Module

This test suite validates the enhanced Sudoku solving capabilities including
constraint propagation, MRV heuristic, and logical solving techniques.
"""
import os
import sys
import pytest
import numpy as np
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.sudsolve import AdvancedSudokuSolver, solve_advanced

@pytest.fixture
def easy_puzzle():
    """Easy Sudoku puzzle for testing."""
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
def hard_puzzle():
    """Hard Sudoku puzzle for testing."""
    return np.array([
        [0, 0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 6, 0, 4, 5],
        [0, 4, 0, 2, 0, 0, 0, 6, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 1, 0, 0]
    ])


@pytest.fixture
def solved_puzzle():
    """Already solved puzzle."""
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
def impossible_puzzle():
    """Impossible puzzle (two 5s in first row)."""
    return np.array([
        [5, 3, 5, 0, 7, 0, 0, 0, 0],  # Two 5s in first row
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ])


class TestAdvancedSudokuSolver:
    """Test cases for the AdvancedSudokuSolver class."""

    def test_solver_initialization(self, easy_puzzle):
        """Test solver initialization."""
        solver = AdvancedSudokuSolver(easy_puzzle)

        assert solver.grid.shape == (9, 9)
        assert solver.candidates.shape == (9, 9)
        assert np.array_equal(solver.grid, easy_puzzle)  # Should be a copy

        # Check that candidates are initialized properly
        for row in range(9):
            for col in range(9):
                if easy_puzzle[row, col] == 0:
                    assert isinstance(solver.candidates[row, col], set)
                    assert len(solver.candidates[row, col]) > 0
                else:
                    assert len(solver.candidates[row, col]) == 0

    def test_get_possible_values(self, easy_puzzle):
        """Test getting possible values for a cell."""
        solver = AdvancedSudokuSolver(easy_puzzle)

        # Test empty cell
        possible = solver._get_possible_values(0, 2)  # First row, third column
        assert isinstance(possible, set)
        assert len(possible) > 0
        assert all(1 <= val <= 9 for val in possible)

        # Test filled cell
        possible_filled = solver._get_possible_values(
            0, 0)  # First row, first column (5)
        assert len(possible_filled) == 0

    def test_assign_value(self, easy_puzzle):
        """Test value assignment."""
        solver = AdvancedSudokuSolver(easy_puzzle)

        # Find an empty cell
        empty_cells = [(r, c) for r in range(9) for c in range(9)
                       if easy_puzzle[r, c] == 0]
        row, col = empty_cells[0]

        original_candidates = solver.candidates[row, col].copy()
        value = list(original_candidates)[0]

        # Assign value
        result = solver._assign_value(row, col, value)

        assert result is True
        assert solver.grid[row, col] == value
        assert len(solver.candidates[row, col]) == 0

    def test_find_naked_singles(self, easy_puzzle):
        """Test finding naked singles."""
        solver = AdvancedSudokuSolver(easy_puzzle)

        # Manually create a situation with a naked single
        test_row, test_col = 0, 2
        solver.candidates[test_row, test_col] = {4}  # Only one possibility

        singles = solver._find_naked_singles()

        # Should find our artificially created single
        assert any(row == test_row and col == test_col and val == 4
                   for row, col, val in singles)

    def test_find_hidden_singles(self, easy_puzzle):
        """Test finding hidden singles."""
        solver = AdvancedSudokuSolver(easy_puzzle)

        singles = solver._find_hidden_singles()

        # Should be a list of tuples
        assert isinstance(singles, list)
        for item in singles:
            assert isinstance(item, tuple)
            assert len(item) == 3
            row, col, val = item
            assert 0 <= row < 9
            assert 0 <= col < 9
            assert 1 <= val <= 9

    def test_solve_with_logic(self, easy_puzzle):
        """Test logic-only solving."""
        solver = AdvancedSudokuSolver(easy_puzzle)
        initial_solved = np.count_nonzero(solver.grid)

        result = solver._solve_with_logic()

        # Should make some progress or detect impossible state
        assert solver.solved_count >= initial_solved

        # Grid should still be valid
        assert solver.grid.shape == (9, 9)
        assert np.all((solver.grid >= 0) & (solver.grid <= 9))

    def test_get_best_cell_mrv(self, easy_puzzle):
        """Test MRV (Most Restricted Variable) heuristic."""
        solver = AdvancedSudokuSolver(easy_puzzle)

        cell = solver._get_best_cell_mrv()

        if cell is not None:  # If there are empty cells
            row, col = cell
            assert 0 <= row < 9
            assert 0 <= col < 9
            assert solver.grid[row, col] == 0
            assert isinstance(solver.candidates[row, col], set)
            assert len(solver.candidates[row, col]) > 0

    def test_state_save_restore(self, easy_puzzle):
        """Test state saving and restoration."""
        solver = AdvancedSudokuSolver(easy_puzzle)

        # Save initial state
        state = solver._save_state()
        original_grid = solver.grid.copy()

        # Modify solver
        if solver.grid[0, 2] == 0:  # If cell is empty
            solver.grid[0, 2] = 7
            solver.solved_count += 1

        # Restore state
        solver._restore_state(state)

        # Should be back to original state
        assert np.array_equal(solver.grid, original_grid)

    def test_full_solve_easy(self, easy_puzzle):
        """Test full solving of easy puzzle."""
        solver = AdvancedSudokuSolver(easy_puzzle)
        result = solver.solve()

        assert result is True

        # Should be a valid solution
        # Check rows
        for row in solver.grid:
            assert len(set(row)) == 9

        # Check columns
        for col in range(9):
            assert len(set(solver.grid[:, col])) == 9

        # Check 3x3 boxes
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = solver.grid[box_row:box_row+3, box_col:box_col+3]
                assert len(set(box.flatten())) == 9

    def test_solve_already_solved(self, solved_puzzle):
        """Test solving an already solved puzzle."""
        solver = AdvancedSudokuSolver(solved_puzzle)
        result = solver.solve()

        assert result is True
        assert np.array_equal(solver.grid, solved_puzzle)

    def test_solve_impossible(self, impossible_puzzle):
        """Test solving an impossible puzzle."""
        solver = AdvancedSudokuSolver(impossible_puzzle)
        result = solver.solve()

        # Should return False for impossible puzzle
        assert result is False

    def test_solve_performance(self, hard_puzzle):
        """Test solving performance on hard puzzle."""
        solver = AdvancedSudokuSolver(hard_puzzle)

        start_time = time.time()
        result = solver.solve()
        solve_time = time.time() - start_time

        # Should solve within reasonable time (10 seconds)
        assert solve_time < 10.0


class TestAdvancedSolverFunctions:
    """Test cases for module-level functions."""

    def test_solve_advanced_basic(self, easy_puzzle):
        """Test basic advanced solving."""
        result = solve_advanced(easy_puzzle)

        assert result is not None
        assert result.shape == (9, 9)
        assert np.all((result >= 1) & (result <= 9))

        # Should be different from original (solved)
        assert not np.array_equal(result, easy_puzzle)

    def test_solve_advanced_invalid_shape(self):
        """Test advanced solving with invalid input shape."""
        invalid_grid = np.zeros((8, 8))
        with pytest.raises(ValueError):
            solve_advanced(invalid_grid)

    def test_benchmark_functionality(self, easy_puzzle):
        """Test basic benchmark functionality."""
        import time

        # Basic timing test
        start_time = time.time()
        solution = solve_advanced(easy_puzzle)
        duration = time.time() - start_time

        assert solution is not None
        assert duration < 1.0  # Should solve quickly

    def test_multiple_dtypes(self, easy_puzzle):
        """Test with different numpy dtypes."""
        # Test with different dtypes
        for dtype in [np.int32, np.int64, np.float32, np.float64]:
            puzzle = easy_puzzle.astype(dtype)
            result = solve_advanced(puzzle)

            assert result is not None
            assert result.shape == (9, 9)
            # Result should be integers regardless of input dtype
            assert np.all(result == result.astype(int))

    def test_consistency_multiple_runs(self, easy_puzzle):
        """Test that solver produces consistent results."""
        results = []

        for _ in range(3):
            result = solve_advanced(easy_puzzle.copy())
            results.append(result)

        # All results should be identical
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i])

    def test_original_grid_preservation(self, easy_puzzle):
        """Test that original grid is not modified."""
        original = easy_puzzle.copy()
        solve_advanced(easy_puzzle)

        # Original should be unchanged
        assert np.array_equal(easy_puzzle, original)


def test_import_availability():
    """Test that all expected components can be imported."""
    from src.sudsolve import (
        AdvancedSudokuSolver, solve_advanced
    )

    # Test that classes and functions exist and are callable
    assert callable(AdvancedSudokuSolver)
    assert callable(solve_advanced)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
