"""
Improved Sudoku solver with multiple predictions and auto-correction features.
"""

import os
import cv2
import numpy as np
import itertools
from typing import List, Tuple, Dict, Optional, Any, Union, Set
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .board_detection import BoardExtractor
from .utils import load_model, plot_grid
# Import SudokuBoard but NOT SudokuSolver from the original module
from .sudoku_solver import SudokuBoard, SudokuSolver


@dataclass
class CellPrediction:
    """Class to store multiple predictions for a single cell."""
    row: int
    col: int
    predictions: List[Tuple[int, float]]  # List of (digit, confidence) pairs
    cell_type: int  # 0=empty, 1=printed, 2=handwritten
    is_problematic: bool = False
    corrected_value: Optional[int] = None
    
    def get_top_prediction(self) -> Tuple[int, float]:
        """Get the top prediction (digit, confidence)."""
        if not self.predictions:
            return (0, 0.0)
        return self.predictions[0]
    
    def get_alternative_predictions(self) -> List[Tuple[int, float]]:
        """Get alternative predictions (all except the top one)."""
        if len(self.predictions) <= 1:
            return []
        return self.predictions[1:]
    
    def __str__(self) -> str:
        """String representation of the cell prediction."""
        top_digit, top_conf = self.get_top_prediction()
        alternatives = self.get_alternative_predictions()
        alt_str = ", ".join([f"{d}({c:.2f})" for d, c in alternatives]) if alternatives else "none"
        
        status = "PROBLEMATIC" if self.is_problematic else ""
        corrected = f" → {self.corrected_value}" if self.corrected_value is not None else ""
        
        return f"Cell({self.row},{self.col}): {top_digit}({top_conf:.2f}) [{alt_str}] {status}{corrected}"


class ImprovedSudokuBoard(SudokuBoard):
    """Enhanced Sudoku board with multiple predictions and correction features."""
    
    def __init__(self, grid: np.ndarray = None, solution: np.ndarray = None):
        """
        Initialize an improved Sudoku board.
        
        Args:
            grid: 9x9 numpy array with the current board state (0 for empty cells)
            solution: 9x9 numpy array with the complete solution (if available)
        """
        super().__init__(grid, solution)
        
        # Store detailed predictions for each cell
        self.cell_predictions = np.empty((9, 9), dtype=object)
        
        # Problematic cells are those that might cause validation issues
        self.problematic_cells = []
        
        # Corrections that have been applied
        self.corrections = []
        
    def set_cell_predictions(self, row: int, col: int, predictions: List[Tuple[int, float]], cell_type: int):
        """
        Set multiple predictions for a cell.
        
        Args:
            row: Row index (0-8)
            col: Column index (0-8)
            predictions: List of (digit, confidence) tuples, sorted by confidence
            cell_type: Cell type (0=empty, 1=printed, 2=handwritten)
        """
        # Create cell prediction object
        cell_pred = CellPrediction(row, col, predictions, cell_type)
        
        # Store in the predictions grid
        self.cell_predictions[row, col] = cell_pred
        
        # Use top prediction to set the grid value
        if predictions:
            top_digit, top_conf = predictions[0]
            self.set_cell(row, col, top_digit, top_conf, cell_type)
            
    def mark_problematic_cell(self, row: int, col: int):
        """
        Mark a cell as problematic (causing validation issues).
        
        Args:
            row: Row index
            col: Column index
        """
        if row < 0 or row > 8 or col < 0 or col > 8:
            return
            
        if self.cell_predictions[row, col] is not None:
            self.cell_predictions[row, col].is_problematic = True
            self.problematic_cells.append((row, col))
            
    def apply_correction(self, row: int, col: int, digit: int):
        """
        Apply a correction to a cell.
        
        Args:
            row: Row index
            col: Column index
            digit: New digit value
        """
        if row < 0 or row > 8 or col < 0 or col > 8:
            return
            
        # Update the grid
        old_value = self.grid[row, col]
        self.grid[row, col] = digit
        
        # Update the cell prediction
        if self.cell_predictions[row, col] is not None:
            self.cell_predictions[row, col].corrected_value = digit
            
        # Track the correction
        self.corrections.append((row, col, old_value, digit))
        
    def find_conflicts(self) -> Dict[Tuple[int, int], List[str]]:
        """
        Find cells that are causing conflicts in the grid.
        
        Returns:
            Dictionary mapping (row, col) to list of conflict reasons
        """
        conflicts = {}
        
        # Check rows for duplicates
        for row in range(9):
            seen = set()
            for col in range(9):
                val = self.grid[row, col]
                if val != 0:
                    if val in seen:
                        # This is a duplicate in the row
                        for c in range(9):
                            if self.grid[row, c] == val:
                                conflicts.setdefault((row, c), []).append(f"row-{row}")
                    seen.add(val)
        
        # Check columns for duplicates
        for col in range(9):
            seen = set()
            for row in range(9):
                val = self.grid[row, col]
                if val != 0:
                    if val in seen:
                        # This is a duplicate in the column
                        for r in range(9):
                            if self.grid[r, col] == val:
                                conflicts.setdefault((r, col), []).append(f"col-{col}")
                    seen.add(val)
        
        # Check 3x3 boxes for duplicates
        for box_row in range(3):
            for box_col in range(3):
                seen = set()
                for i in range(3):
                    for j in range(3):
                        row, col = box_row*3 + i, box_col*3 + j
                        val = self.grid[row, col]
                        if val != 0:
                            if val in seen:
                                # This is a duplicate in the box
                                for di in range(3):
                                    for dj in range(3):
                                        r, c = box_row*3 + di, box_col*3 + dj
                                        if self.grid[r, c] == val:
                                            conflicts.setdefault((r, c), []).append(f"box-{box_row}{box_col}")
                            seen.add(val)
        
        return conflicts
    
    def identify_problematic_cells(self):
        """
        Identify cells that are causing validation issues.
        """
        # Reset problematic cells
        self.problematic_cells = []
        
        # Find conflicts
        conflicts = self.find_conflicts()
        
        # Mark conflicting cells as problematic
        for (row, col), reasons in conflicts.items():
            self.mark_problematic_cell(row, col)
            
    def validate_with_improvements(self) -> bool:
        """
        Validate the grid and identify problematic cells.
        
        Returns:
            bool: True if the grid is valid, False otherwise
        """
        is_valid = self.validate()
        
        if not is_valid:
            # Identify problematic cells
            self.identify_problematic_cells()
            
        return is_valid
    
    def auto_correct_board(self, max_iterations: int = 5) -> bool:
        """
        Attempt to automatically correct the board by trying alternative predictions.
        
        Args:
            max_iterations: Maximum number of iterations to try
            
        Returns:
            bool: True if the board could be corrected to be valid
        """
        # Start from current state
        original_grid = self.grid.copy()
        best_grid = original_grid.copy()
        iteration = 0
        
        # First, make sure we've identified problematic cells
        if not self.problematic_cells:
            self.validate_with_improvements()
            
        if not self.problematic_cells:
            # No problematic cells, board is already valid
            return True
            
        # Keep track of the best attempt
        best_valid = False
        best_problematic_count = len(self.problematic_cells)
        
        # Try to correct the board
        while iteration < max_iterations and not best_valid:
            # Get a list of problematic cells with alternatives
            cells_with_alternatives = []
            for row, col in self.problematic_cells:
                cell_pred = self.cell_predictions[row, col]
                if cell_pred and len(cell_pred.predictions) > 1:
                    cells_with_alternatives.append((row, col))
            
            if not cells_with_alternatives:
                # No cells with alternatives to try
                break
                
            # Try combinations of alternative digits
            # Start with changing just one cell, then try more if needed
            for num_cells_to_change in range(1, min(4, len(cells_with_alternatives))+1):
                for cells_to_try in itertools.combinations(cells_with_alternatives, num_cells_to_change):
                    # Generate alternative combinations
                    alternatives = []
                    for row, col in cells_to_try:
                        cell_pred = self.cell_predictions[row, col]
                        # Use all predictions except the top one that's already been tried
                        alt_preds = [(i, pred[0], pred[1]) for i, pred in enumerate(cell_pred.predictions[1:], 1)]
                        alternatives.append(alt_preds)
                    
                    # Try each combination of alternatives
                    for alt_combo in itertools.product(*alternatives):
                        # Apply this combination
                        self.grid = original_grid.copy()
                        
                        # Apply each alternative
                        for i, combo in enumerate(alt_combo):
                            pred_idx, digit, confidence = combo
                            row, col = cells_to_try[i]
                            self.grid[row, col] = digit
                            
                        # Check if this improves the board
                        is_valid = self.validate()
                        if is_valid:
                            # We found a valid solution!
                            best_valid = True
                            best_grid = self.grid.copy()
                            
                            # Record corrections
                            for i, combo in enumerate(alt_combo):
                                pred_idx, digit, confidence = combo
                                row, col = cells_to_try[i]
                                self.apply_correction(row, col, digit)
                                
                            # We're done
                            return True
                            
                        # Count problematic cells in this attempt
                        self.identify_problematic_cells()
                        problematic_count = len(self.problematic_cells)
                        
                        if problematic_count < best_problematic_count:
                            # This attempt has fewer problematic cells
                            best_problematic_count = problematic_count
                            best_grid = self.grid.copy()
            
            # Move to next iteration
            iteration += 1
            
        # Restore best attempt
        self.grid = best_grid.copy()
        self.validate_with_improvements()
        
        return best_valid
    
    def display_with_alternatives(self, title: str = "Sudoku Board with Alternatives"):
        """
        Display the board with alternative predictions shown.
        
        Args:
            title: Plot title
        """
        plt.figure(figsize=(12, 12))
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
                        
                    # Set special color for corrected cells or problematic cells
                    cell_pred = self.cell_predictions[i, j]
                    if cell_pred:
                        if cell_pred.corrected_value is not None:
                            # This cell has been corrected
                            color = 'green'
                        elif cell_pred.is_problematic:
                            # This cell is problematic
                            color = 'red'
                    
                    # Show the digit
                    plt.text(j + 0.5, i + 0.5, str(self.grid[i, j]),
                           ha='center', va='center',
                           color=color,
                           fontsize=15,
                           fontweight='bold')
                    
                    # For corrected cells, show the original value -> new value
                    if cell_pred and cell_pred.corrected_value is not None:
                        # Get the original prediction
                        original_value = cell_pred.predictions[0][0]  # First prediction's value
                        # Show the correction at the bottom of the cell
                        plt.text(j + 0.5, i + 0.85, f"{original_value}→{cell_pred.corrected_value}",
                               ha='center', va='center',
                               color=color,
                               fontsize=8,
                               fontweight='bold')
                    # Show alternatives if present and not corrected
                    elif cell_pred and len(cell_pred.predictions) > 1 and cell_pred.corrected_value is None:
                        # Get top 3 alternatives
                        alts = cell_pred.predictions[1:4]  # Up to 3 alternatives
                        if alts:
                            alt_text = " ".join([f"{d}" for d, _ in alts])
                            plt.text(j + 0.5, i + 0.8, alt_text,
                                   ha='center', va='center',
                                   color=color,
                                   fontsize=8)
                
                # For empty cells, show top alternatives if confidence is high enough
                elif self.cell_predictions[i, j] is not None:
                    cell_pred = self.cell_predictions[i, j]
                    if cell_pred.predictions:
                        top_pred = cell_pred.predictions[0]
                        if top_pred[1] > 0.1:  # Minimum confidence threshold
                            plt.text(j + 0.5, i + 0.5, f"({top_pred[0]}?)",
                                   ha='center', va='center',
                                   color='lightgray',
                                   fontsize=10)

        plt.xlim(0, 9)
        plt.ylim(9, 0)
        plt.title(title)
        plt.axis('equal')
        plt.axis('off')
        plt.tight_layout()
        plt.show()


class ImprovedSudokuProcessor:
    """Enhanced processor for Sudoku images with multiple predictions and auto-correction."""
    
    def __init__(self, 
                 digit_model_path: Optional[str] = None,
                 cell_type_model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 top_n: int = 3,
                 models_dir: Optional[str] = None):
        """
        Initialize the improved processor.
        
        Args:
            digit_model_path: Path to the saved digit recognition model
            cell_type_model_path: Path to the saved cell type classification model
            confidence_threshold: Minimum confidence threshold for accepting predictions
            top_n: Number of top predictions to keep for each cell
            models_dir: Directory containing models (if model paths not specified)
        """
        from .utils import setup_project_paths
        
        # Set up paths
        paths = setup_project_paths()
        models_dir = models_dir or paths['models_dir']
        
        # Initialize board extractor
        self.board_extractor = BoardExtractor()
        self.top_n = top_n
        
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
                      display_steps: bool = False,
                      auto_correct: bool = True) -> Optional[ImprovedSudokuBoard]:
        """
        Process a Sudoku image and extract a digital board with multiple predictions.
        
        Args:
            image_path: Path to the image
            display_steps: Whether to display intermediate steps
            auto_correct: Whether to attempt automatic correction of invalid boards
            
        Returns:
            ImprovedSudokuBoard object or None if processing failed
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
            
        # Create a new improved Sudoku board
        board = ImprovedSudokuBoard()
        
        # Process each cell
        for cell_img, (row, col) in cells:
            self.process_cell_with_alternatives(cell_img, row, col, board)
            
        # Update board mode and validate
        board.update_mode()
        is_valid = board.validate_with_improvements()
        
        if not is_valid and auto_correct:
            print("Board is invalid. Attempting automatic correction...")
            corrected = board.auto_correct_board()
            if corrected:
                print("Successfully corrected the board!")
            else:
                print("Could not fully correct the board. Some issues remain.")
                
        return board
    
    def process_cell_with_alternatives(self, cell_img: np.ndarray, row: int, col: int, board: ImprovedSudokuBoard):
        """
        Process a single cell and store multiple predictions.
        
        Args:
            cell_img: Cell image
            row: Row index
            col: Column index
            board: ImprovedSudokuBoard to update
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
        digit_preds = self.digit_model.predict(model_input, verbose=0)[0]
        cell_type_preds = self.cell_type_model.predict(model_input, verbose=0)[0]
        
        # Get top N digit predictions with confidence
        digit_indices = np.argsort(digit_preds)[::-1][:self.top_n]  # Top N indices
        digit_predictions = [(int(idx), float(digit_preds[idx])) for idx in digit_indices]
        
        # Get top cell type prediction
        cell_type_class = np.argmax(cell_type_preds)
        cell_type_conf = cell_type_preds[cell_type_class]
        
        # Handle empty cells specially
        # If the cell is predicted as empty (cell_type=0) or the top digit is 0 with high confidence,
        # we should set the cell value to 0 (empty)
        if cell_type_class == 0 or (digit_predictions[0][0] == 0 and digit_predictions[0][1] > 0.8):
            # This is an empty cell
            # Still store all predictions, but mark the cell as empty
            board.set_cell_predictions(row, col, digit_predictions, 0)
        else:
            # Non-empty cell - store all predictions
            board.set_cell_predictions(row, col, digit_predictions, cell_type_class)
    
    def solve_board(self, board: ImprovedSudokuBoard, timeout_seconds: float = 10.0) -> bool:
        """
        Solve a Sudoku board with timeout.
        
        Args:
            board: ImprovedSudokuBoard to solve
            timeout_seconds: Maximum time to spend trying to solve (in seconds)
            
        Returns:
            bool: True if a solution was found
        """
        return SudokuSolver.solve(board)
        
    def print_cell_predictions(self, board: ImprovedSudokuBoard, min_confidence: float = 0.3):
        """
        Print detailed predictions for cells.
        
        Args:
            board: ImprovedSudokuBoard with predictions
            min_confidence: Minimum confidence to show alternative predictions
        """
        print("Cell Predictions (with alternatives):")
        for row in range(9):
            for col in range(9):
                if board.cell_predictions[row, col] is not None:
                    cell_pred = board.cell_predictions[row, col]
                    
                    # Filter predictions by confidence
                    predictions = [p for p in cell_pred.predictions if p[1] >= min_confidence]
                    
                    if predictions:
                        cell_pred.predictions = predictions
                        print(cell_pred)
        
        print(f"\nProblematic cells: {len(board.problematic_cells)}")
        for row, col in board.problematic_cells:
            if board.cell_predictions[row, col] is not None:
                print(f"  {board.cell_predictions[row, col]}")