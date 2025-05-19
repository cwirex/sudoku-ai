"""
Sudoku image processing, validation, and solving functionality.
"""

import os
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional, Any, Union
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

from .board_detection import BoardExtractor
from .utils import load_model, plot_grid



class SudokuBoard:
    """Class representing a Sudoku board with its state and metadata."""
    
    def __init__(self, grid: np.ndarray = None, solution: np.ndarray = None):
        """
        Initialize a Sudoku board.
        
        Args:
            grid: 9x9 numpy array with the current board state (0 for empty cells)
            solution: 9x9 numpy array with the complete solution (if available)
        """
        # Initialize with empty 9x9 grid if none provided
        self.grid = grid if grid is not None else np.zeros((9, 9), dtype=int)
        self.solution = solution
        self.handwritten_mask = np.zeros((9, 9), dtype=int)
        self.cell_types = np.zeros((9, 9), dtype=int)  # 0=empty, 1=printed, 2=handwritten
        self.confidence = np.zeros((9, 9), dtype=float)
        self.is_valid = None  # Will be set after validation
        self.mode = "unknown"  # "printed", "handwritten", or "mixed"
        
    def set_cell(self, row: int, col: int, value: int, confidence: float = 1.0, cell_type: int = 1):
        """
        Set a cell value with metadata.
        
        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            value: Cell value (0-9, where 0 is empty)
            confidence: Confidence score for the prediction (0-1)
            cell_type: Cell type (0=empty, 1=printed, 2=handwritten)
        """
        self.grid[row, col] = value
        self.confidence[row, col] = confidence
        self.cell_types[row, col] = cell_type
        
        # Update handwritten mask if applicable
        if cell_type == 2:
            self.handwritten_mask[row, col] = 1
            
    def is_complete(self) -> bool:
        """Check if the grid is completely filled (no zeros)."""
        return np.all(self.grid != 0)
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get a list of empty cell coordinates (row, col)."""
        empty_indices = np.where(self.grid == 0)
        return list(zip(empty_indices[0], empty_indices[1]))
    
    def validate(self) -> bool:
        """
        Validate the current grid state for Sudoku rules.
        
        Returns:
            bool: True if the grid is valid, False otherwise
        """
        # Check each row
        for row in range(9):
            # Filter out zeros (empty cells)
            filled = self.grid[row, :][self.grid[row, :] != 0]
            if len(filled) != len(set(filled)):
                self.is_valid = False
                return False
        
        # Check each column
        for col in range(9):
            filled = self.grid[:, col][self.grid[:, col] != 0]
            if len(filled) != len(set(filled)):
                self.is_valid = False
                return False
        
        # Check each 3x3 box
        for box_row in range(3):
            for box_col in range(3):
                # Extract the 3x3 box
                box = self.grid[box_row*3:(box_row+1)*3, box_col*3:(box_col+1)*3]
                filled = box[box != 0]
                if len(filled) != len(set(filled)):
                    self.is_valid = False
                    return False
        
        self.is_valid = True
        return True
    
    def update_mode(self):
        """Update the board mode based on cell types."""
        if not np.any(self.grid > 0):
            self.mode = "unknown"
            return
            
        # Count non-empty cell types
        cells_with_digits = np.sum(self.grid > 0)
        handwritten_cells = np.sum(self.handwritten_mask)
        printed_cells = np.sum((self.grid > 0) & (self.handwritten_mask == 0))
        
        if handwritten_cells == 0 and printed_cells > 0:
            self.mode = "printed"
        elif printed_cells == 0 and handwritten_cells > 0:
            self.mode = "handwritten"
        else:
            self.mode = "mixed"
    
    def get_stats(self) -> Dict:
        """Get statistics about the board."""
        total_filled = np.sum(self.grid > 0)
        total_empty = np.sum(self.grid == 0)
        handwritten = np.sum(self.handwritten_mask)
        printed = total_filled - handwritten
        
        # Calculate average confidence for non-empty cells
        avg_confidence = 0
        if total_filled > 0:
            avg_confidence = np.sum(self.confidence[self.grid > 0]) / total_filled
        
        return {
            "total_cells": 81,
            "filled_cells": int(total_filled),
            "empty_cells": int(total_empty),
            "handwritten_cells": int(handwritten),
            "printed_cells": int(printed),
            "fill_percentage": f"{total_filled/81:.2%}",
            "is_valid": self.is_valid,
            "mode": self.mode,
            "average_confidence": f"{avg_confidence:.2%}"
        }
    
    def display(self, title: str = "Sudoku Board", show_confidence: bool = False, show_cell_types: bool = False):
        """
        Display the board with optional metadata.
        
        Args:
            title: Plot title
            show_confidence: Whether to show confidence scores
            show_cell_types: Whether to show cell types (printed/handwritten)
        """
        plt.figure(figsize=(10, 10))
        plt.fill([0, 9, 9, 0], [0, 0, 9, 9], 'white')

        # Draw grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            plt.axhline(y=i, color='black', linewidth=lw)
            plt.axvline(x=i, color='black', linewidth=lw)

        # Add numbers and metadata
        for i in range(9):
            for j in range(9):
                if self.grid[i, j] != 0:
                    # Set colors based on cell type
                    if self.handwritten_mask[i, j] == 1:
                        color = 'blue'  # Handwritten
                    else:
                        color = 'black'  # Printed
                    
                    plt.text(j + 0.5, i + 0.5, str(self.grid[i, j]),
                           ha='center', va='center',
                           color=color,
                           fontsize=15,
                           fontweight='bold')
                    
                    # Show confidence if requested
                    if show_confidence:
                        confidence_str = f"{self.confidence[i, j]:.2f}"
                        plt.text(j + 0.5, i + 0.8, confidence_str,
                               ha='center', va='center',
                               color=color,
                               fontsize=8)
                    
                    # Show cell type if requested
                    if show_cell_types:
                        cell_type_str = "H" if self.handwritten_mask[i, j] == 1 else "P"
                        plt.text(j + 0.2, i + 0.2, cell_type_str,
                               ha='center', va='center',
                               color=color,
                               fontsize=8)

        plt.xlim(0, 9)
        plt.ylim(9, 0)
        plt.title(title)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()
        
    def print_grid(self):
        """Print a text representation of the grid."""
        print("┌───────┬───────┬───────┐")
        for i in range(9):
            line = "│ "
            for j in range(9):
                if self.grid[i, j] == 0:
                    line += ". "
                else:
                    line += str(self.grid[i, j]) + " "
                if j % 3 == 2 and j < 8:
                    line += "│ "
            line += "│"
            print(line)
            if i % 3 == 2 and i < 8:
                print("├───────┼───────┼───────┤")
        print("└───────┴───────┴───────┘")




class SudokuSolver:
    """Class for solving Sudoku puzzles with timeout and validation checks."""
    
    @staticmethod
    def is_valid_move(board: np.ndarray, row: int, col: int, num: int) -> bool:
        """
        Optimized validity check using NumPy vectorization.
        
        Args:
            board: Current board state
            row: Row index
            col: Column index
            num: Number to place
            
        Returns:
            bool: True if the move is valid
        """
        # Check row and column
        if np.any(board[row, :] == num) or np.any(board[:, col] == num):
            return False
            
        # Check 3x3 box
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        if np.any(board[box_row:box_row+3, box_col:box_col+3] == num):
            return False
            
        return True
    
    @staticmethod
    def is_valid_board(grid: np.ndarray) -> bool:
        """
        Validate the entire board before solving.
        
        Args:
            grid: Sudoku grid to validate
            
        Returns:
            bool: True if the board is valid
        """
        for i in range(9):
            # Check row
            row = grid[i, :][grid[i, :] != 0]
            if len(row) != len(np.unique(row)):
                return False
                
            # Check column
            col = grid[:, i][grid[:, i] != 0]
            if len(col) != len(np.unique(col)):
                return False
                
        # Check 3x3 boxes
        for i in range(0, 9, 3):
            for j in range(0, 9, 3):
                box = grid[i:i+3, j:j+3].flatten()
                box = box[box != 0]
                if len(box) != len(np.unique(box)):
                    return False
                    
        return True
    
    @staticmethod
    def find_empty_cell(grid: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Find the empty cell with fewest possibilities (MRV heuristic).
        
        Args:
            grid: Sudoku grid
            
        Returns:
            Tuple of (row, col) or None if no empty cells
        """
        empty_cells = np.argwhere(grid == 0)
        if len(empty_cells) == 0:
            return None
            
        min_options = 10
        best_cell = None
        
        for row, col in empty_cells:
            options = sum(
                1 for num in range(1, 10) 
                if SudokuSolver.is_valid_move(grid, row, col, num)
            )
            if options < min_options:
                min_options = options
                best_cell = (row, col)
                if min_options == 1:  # Early exit if found cell with only 1 option
                    break
                    
        return best_cell
    
    @staticmethod
    def _solve_recursive(grid: np.ndarray, start_time: float, timeout: float) -> bool:
        """
        Recursive solver with timeout check.
        
        Args:
            grid: Current grid state
            start_time: Timestamp when solving began
            timeout: Maximum allowed duration in seconds
            
        Returns:
            bool: True if solved successfully
        """
        if time.time() - start_time > timeout:
            raise TimeoutError()
            
        cell = SudokuSolver.find_empty_cell(grid)
        if not cell:
            return True
            
        row, col = cell
        
        for num in range(1, 10):
            if SudokuSolver.is_valid_move(grid, row, col, num):
                grid[row, col] = num
                if SudokuSolver._solve_recursive(grid, start_time, timeout):
                    return True
                grid[row, col] = 0
                
        return False
    
    @staticmethod
    def solve(board: SudokuBoard, validate_first: bool = True, timeout_sec: float = 10) -> bool:
        """
        Solve with validation and timeout handling.
        
        Args:
            board: SudokuBoard to solve
            validate_first: Whether to validate before solving
            timeout_sec: Total timeout in seconds (includes validation)
            
        Returns:
            bool: True if solved successfully
        """
        start_time = time.time()
        working_grid = board.grid.copy()
        
        def solve_with_timeout():
            return SudokuSolver._solve_recursive(
                working_grid, 
                start_time, 
                timeout_sec
            )
        
        try:
            # Phase 1: Validation (3 sec max)
            if validate_first:
                with ThreadPoolExecutor() as executor:
                    validation_future = executor.submit(SudokuSolver.is_valid_board, working_grid)
                    is_valid = validation_future.result(timeout=3)
                    
                if not is_valid:
                    return False
                    
            # Phase 2: Solving (remaining time)
            remaining_time = timeout_sec - (time.time() - start_time)
            if remaining_time <= 0:
                return False
                
            with ThreadPoolExecutor() as executor:
                solving_future = executor.submit(solve_with_timeout)
                solved = solving_future.result(timeout=remaining_time)
                
                if solved:
                    board.solution = working_grid
                    return True
                return False
                
        except TimeoutError:
            return False
        except Exception:
            return False

class SudokuProcessor:
    """Main class for processing Sudoku images into digital boards."""
    
    def __init__(self, 
                 digit_model_path: Optional[str] = None,
                 cell_type_model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 models_dir: Optional[str] = None):
        """
        Initialize the processor with detection and recognition models.
        
        Args:
            digit_model_path: Path to the saved digit recognition model
            cell_type_model_path: Path to the saved cell type classification model
            confidence_threshold: Minimum confidence threshold for accepting predictions
            models_dir: Directory containing models (if model paths not specified)
        """
        from .utils import setup_project_paths
        
        # Set up paths
        paths = setup_project_paths()
        models_dir = models_dir or paths['models_dir']
        
        # Initialize board extractor
        self.board_extractor = BoardExtractor()
        
        # Load models
        try:
            if digit_model_path:
                self.digit_model = load_model(digit_model_path)
            else:
                # Try to find a digit model in the models directory
                for filename in os.listdir(models_dir):
                    if 'digit' in filename.lower() and filename.endswith('.h5'):
                        self.digit_model = load_model(os.path.splitext(filename)[0], models_dir)
                        print(f"Found and loaded digit model: {filename}")
                        break
                else:
                    print("No digit model found. Please provide a path to a digit recognition model.")
                    self.digit_model = None
            
            if cell_type_model_path:
                self.cell_type_model = load_model(cell_type_model_path)
            else:
                # Try to find a cell type model in the models directory
                for filename in os.listdir(models_dir):
                    if 'cell' in filename.lower() and filename.endswith('.h5'):
                        self.cell_type_model = load_model(os.path.splitext(filename)[0], models_dir)
                        print(f"Found and loaded cell type model: {filename}")
                        break
                else:
                    print("No cell type model found. Please provide a path to a cell type classification model.")
                    self.cell_type_model = None
        except Exception as e:
            print(f"Error loading models: {e}")
            self.digit_model = None
            self.cell_type_model = None
            
        self.confidence_threshold = confidence_threshold
    
    def process_image(self, 
                      image_path: str, 
                      display_steps: bool = False) -> Optional[SudokuBoard]:
        """
        Process a Sudoku image and extract a digital board.
        
        Args:
            image_path: Path to the image
            display_steps: Whether to display intermediate steps
            
        Returns:
            SudokuBoard object or None if processing failed
        """
        # Read the image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image from {image_path}")
                return None
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            # Assume it's already a numpy array
            image = image_path
            
        # Display the original image
        if display_steps:
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            plt.title("Original Image")
            plt.axis('off')
            plt.show()
            
        # Extract the board and cells
        warped, cells = self.board_extractor.extract_board(image, display_steps=display_steps)
        
        if warped is None or cells is None:
            print("Board extraction failed.")
            return None
            
        # Create a new Sudoku board
        board = SudokuBoard()
        
        # Process each cell
        for cell_img, (row, col) in cells:
            self.process_cell(cell_img, row, col, board)
            
        # Update board mode and validate
        board.update_mode()
        board.validate()
        
        return board
    
    def process_cell(self, cell_img: np.ndarray, row: int, col: int, board: SudokuBoard):
        """
        Process a single cell and update the board.
        
        Args:
            cell_img: Cell image
            row: Row index
            col: Column index
            board: SudokuBoard to update
        """
        if self.digit_model is None or self.cell_type_model is None:
            print("Models not loaded. Cannot process cell.")
            return
            
        # Resize cell image to match model input
        resized_cell = cv2.resize(cell_img, (28, 28))
        
        # Convert to grayscale
        if len(resized_cell.shape) == 3:
            gray_cell = cv2.cvtColor(resized_cell, cv2.COLOR_RGB2GRAY)
        else:
            gray_cell = resized_cell
            
        # Normalize
        norm_cell = gray_cell / 255.0
        
        # Prepare for model input [1, height, width, 1]
        model_input = norm_cell.reshape(1, 28, 28, 1)
        
        # Predict digit and cell type
        digit_pred = self.digit_model.predict(model_input, verbose=0)[0]
        cell_type_pred = self.cell_type_model.predict(model_input, verbose=0)[0]
        
        # Get predicted classes and confidences
        digit_class = np.argmax(digit_pred)
        digit_conf = digit_pred[digit_class]
        
        cell_type_class = np.argmax(cell_type_pred)
        cell_type_conf = cell_type_pred[cell_type_class]
        
        # Set the cell in the board
        # Only set non-empty cells with sufficient confidence
        if cell_type_class == 0 or (digit_class == 0 and digit_conf > 0.8):
            # Empty cell
            board.set_cell(row, col, 0, 1.0, 0)
        elif digit_conf >= self.confidence_threshold:
            # Non-empty cell with sufficient confidence
            board.set_cell(row, col, digit_class, digit_conf, cell_type_class)
        else:
            # Low confidence, set as empty
            board.set_cell(row, col, 0, digit_conf, 0)
    
    def solve_board(self, board: SudokuBoard) -> bool:
        """
        Solve a Sudoku board.
        
        Args:
            board: SudokuBoard to solve
            
        Returns:
            bool: True if a solution was found
        """
        return SudokuSolver.solve(board)