
import random

import numpy as np
from src.config.generator_config import SudokuGeneratorConfig


class SudokuPuzzleGenerator:
    """Generates valid Sudoku puzzles and solutions."""

    def __init__(self, config=None):
        """Initialize with configuration."""
        self.config = config or SudokuGeneratorConfig()

    def generate(self, mask_rate=None):
        """
        Generate a Sudoku puzzle and its solution.

        Args:
            mask_rate: Percentage of cells to leave empty (0.0 to 1.0)
                      If None, a random value will be chosen from config

        Returns:
            grid: 9x9 numpy array with the puzzle (0 for empty cells)
            solution: 9x9 numpy array with the full solution
        """
        if mask_rate is None:
            mask_rate = random.choice(self.config.mask_rates)

        # Create a base pattern (this is a valid Sudoku solution)
        base = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]

        # Randomize the solution by shuffling rows and columns within blocks
        solution = np.array(base, dtype=int)

        # Shuffle rows within each block
        for block in range(3):
            rows = range(block * 3, (block + 1) * 3)
            perm = list(rows)
            random.shuffle(perm)
            solution[[rows[0], rows[1], rows[2]]] = solution[[perm[0], perm[1], perm[2]]]

        # Shuffle columns within each block
        for block in range(3):
            cols = range(block * 3, (block + 1) * 3)
            perm = list(cols)
            random.shuffle(perm)
            solution[:, [cols[0], cols[1], cols[2]]] = solution[:, [perm[0], perm[1], perm[2]]]

        # Create a masked version as the puzzle grid
        grid = solution.copy()
        mask = np.random.random((9, 9)) < mask_rate
        grid[mask] = 0

        return grid, solution
