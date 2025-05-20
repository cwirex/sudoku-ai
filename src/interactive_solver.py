"""
Interactive Sudoku solver with comprehensive user interface and guidance.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Any, Union, Set
from dataclasses import dataclass

from .improved_solver import ImprovedSudokuProcessor, ImprovedSudokuBoard
from .board_detection import BoardExtractor
from .sudoku_solver import SudokuSolver
from .utils import plot_grid


@dataclass
class HintSuggestion:
    """Class to represent a hint suggestion."""
    row: int
    col: int
    value: int
    reason: str
    difficulty: str  # "easy", "medium", "hard"


class InteractiveSudokuSolver:
    """Interactive Sudoku solver with comprehensive user interface."""
    
    def __init__(self, 
                 digit_model_path: Optional[str] = None,
                 cell_type_model_path: Optional[str] = None,
                 confidence_threshold: float = 0.5,
                 models_dir: Optional[str] = None):
        """
        Initialize the interactive solver.
        
        Args:
            digit_model_path: Path to the saved digit recognition model
            cell_type_model_path: Path to the saved cell type classification model
            confidence_threshold: Minimum confidence threshold for accepting predictions
            models_dir: Directory containing models (if model paths not specified)
        """
        # Initialize the improved processor
        self.processor = ImprovedSudokuProcessor(
            digit_model_path=digit_model_path,
            cell_type_model_path=cell_type_model_path,
            confidence_threshold=confidence_threshold,
            models_dir=models_dir
        )
        
        # Initialize board extractor for detection visualization
        self.board_extractor = BoardExtractor()
        
        # State variables
        self.current_board = None
        self.original_image = None
        self.extracted_board_image = None
        self.problematic_positions = []
        self.solution = None
        self.last_hint = None
        self.applied_hints = []  # Track applied hints
        
    def process_image(self, image_path: str, display_results: bool = True) -> bool:
        """
        Process a Sudoku image and initialize the interactive session.
        
        Args:
            image_path: Path to the image
            display_results: Whether to display detection results
            
        Returns:
            bool: True if processing was successful
        """
        # Load the image
        if isinstance(image_path, str):
            self.original_image = cv2.imread(image_path)
            if self.original_image is None:
                print(f"Could not read image from {image_path}")
                return False
            self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        else:
            self.original_image = image_path
            
        # Process with the improved processor
        self.current_board = self.processor.process_image(
            self.original_image, 
            display_steps=False, 
            auto_correct=True
        )
        
        if self.current_board is None:
            print("Failed to process the image. Could not extract board or recognize digits.")
            return False
            
        # Extract the board image for display
        warped, _ = self.board_extractor.extract_board(self.original_image, display_steps=False)
        self.extracted_board_image = warped
        
        if display_results:
            self.show_detection_results()
            
        return True
    
    def show_detection_results(self):
        """Show the detection and extraction results."""
        if self.original_image is None or self.current_board is None:
            print("No image processed yet. Call process_image() first.")
            return
            
        # Show original image with detected corners
        self._show_detection_overlay()
        
        # Show side-by-side comparison
        self._show_extraction_comparison()
        
        # Show current board status
        self._show_board_status()
    
    def _show_detection_overlay(self):
        """Show original image with detected board outline."""
        # Extract corners for visualization
        processed = self.board_extractor.preprocess_image(self.original_image, show_steps=False)
        corners = self.board_extractor.find_board_corners(processed, self.original_image, show_steps=False)
        
        if corners is not None:
            # Create visualization
            img_with_corners = self.original_image.copy()
            
            # Draw the detected board outline
            corner_points = np.array(corners, dtype=np.int32)
            cv2.polylines(img_with_corners, [corner_points], True, (0, 255, 0), 3)
            
            # Draw corner points
            for i, corner in enumerate(corners):
                cv2.circle(img_with_corners, tuple(corner), 8, (255, 0, 0), -1)
                cv2.putText(img_with_corners, str(i+1), 
                           (corner[0]-10, corner[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(img_with_corners)
            plt.title("Original Image with Detected Board (Green outline, Red corner points)")
            plt.axis('off')
            plt.show()
        else:
            print("Could not detect board corners for visualization.")
    
    def _show_extraction_comparison(self):
        """Show side-by-side comparison of extracted board and digital version."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left: Extracted board image
        if self.extracted_board_image is not None:
            ax1.imshow(self.extracted_board_image)
            ax1.set_title("Extracted Board (Transformed)")
            ax1.axis('off')
        else:
            ax1.text(0.5, 0.5, "Board extraction failed", 
                    ha='center', va='center', transform=ax1.transAxes)
            ax1.set_title("Extracted Board")
        
        # Right: Digital version
        self._show_digital_board(ax2, title="Digital Board (After Auto-Correction)")
        
        plt.tight_layout()
        plt.show()
    
    def _show_digital_board(self, ax, title="Digital Board", highlight_problematic=False):
        """Show the digital version of the board with color coding."""
        ax.fill([0, 9, 9, 0], [0, 0, 9, 9], 'white')

        # Draw grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            ax.axhline(y=i, color='black', linewidth=lw)
            ax.axvline(x=i, color='black', linewidth=lw)

        # Add numbers with color coding
        for i in range(9):
            for j in range(9):
                if self.current_board.grid[i, j] != 0:
                    # Determine color based on cell type and status
                    color = 'black'  # Default for printed
                    
                    # Check if this is an applied hint
                    if (i, j) in self.applied_hints:
                        color = 'darkgreen'  # Applied hints
                    elif self.current_board.handwritten_mask[i, j] == 1:
                        if highlight_problematic and (i, j) in self.problematic_positions:
                            color = 'red'  # Problematic handwritten
                        else:
                            color = 'darkblue'  # Normal handwritten
                    
                    ax.text(j + 0.5, i + 0.5, str(self.current_board.grid[i, j]),
                           ha='center', va='center',
                           color=color,
                           fontsize=15,
                           fontweight='bold')

        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_title(title)
        ax.axis('equal')
        ax.axis('off')
    
    def _show_board_status(self):
        """Show current board status and statistics."""
        stats = self.current_board.get_stats()
        
        print("\n" + "="*50)
        print("BOARD STATUS")
        print("="*50)
        
        for key, value in stats.items():
            print(f"{key.replace('_', ' ').title()}: {value}")
        
        # Show validity and next steps
        if self.current_board.is_valid:
            print("\n✅ Board is VALID!")
            print("\nNext steps:")
            print("1. Get a hint for the next move")
            print("2. Show the complete solution")
        else:
            print("\n❌ Board is INVALID!")
            # Identify problematic handwritten cells
            self._identify_problematic_cells()
            
            if self.problematic_positions:
                print(f"\nFound {len(self.problematic_positions)} possibly incorrect handwritten digits.")
                print("\nNext steps:")
                print("1. View problematic cells")
                print("2. Manually correct specific cells")
            else:
                print("\nNo specific problematic handwritten cells identified.")
                print("The board may have printed digit recognition errors or other issues.")
        
        print("="*50)
    
    def _identify_problematic_cells(self):
        """Identify handwritten cells that are causing validation issues."""
        self.problematic_positions = []
        
        if not hasattr(self.current_board, 'find_conflicts'):
            return
            
        # Get all conflicts
        conflicts = self.current_board.find_conflicts()
        
        # Filter to only handwritten cells
        for (row, col), reasons in conflicts.items():
            if self.current_board.handwritten_mask[row, col] == 1:
                self.problematic_positions.append((row, col))
    
    def show_problematic_cells(self):
        """Show the board with problematic cells highlighted in red."""
        if not self.problematic_positions:
            print("No problematic cells identified.")
            return
            
        plt.figure(figsize=(10, 10))
        ax = plt.gca()
        
        self._show_digital_board(
            ax, 
            title=f"Board with {len(self.problematic_positions)} Problematic Handwritten Cells (Red)",
            highlight_problematic=True
        )
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                      markersize=10, label='Printed Digits'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                      markersize=10, label='Handwritten Digits'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                      markersize=10, label='Problematic Handwritten')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show()
        
        # List the problematic positions
        print(f"\nProblematic positions (row, col):")
        for i, (row, col) in enumerate(self.problematic_positions, 1):
            value = self.current_board.grid[row, col]
            print(f"{i}. Position ({row+1}, {col+1}): {value}")  # 1-based indexing for display
    
    def manual_cell_correction(self):
        """Allow user to manually correct specific cells."""
        if self.current_board is None:
            print("No board loaded.")
            return
            
        print("\nManual Cell Correction")
        print("Enter corrections in the format 'row,col,value' (e.g., '2,3,5')")
        print("Enter multiple corrections separated by semicolons (e.g., '2,3,5; 4,5,9')")
        print("Row and column indices are 1-9")
        print("Enter 'done' when finished, 'show' to display current board")
        
        while True:
            # Show current board
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            self._show_digital_board(ax, title="Current Board State")
            plt.show()
            
            user_input = input("\nCorrections (or 'done'/'show'): ").strip()
            
            if user_input.lower() == 'done':
                break
            elif user_input.lower() == 'show':
                continue
            
            # Parse and apply corrections
            corrections = [c.strip() for c in user_input.split(';')]
            corrections_applied = False
            
            for correction in corrections:
                try:
                    parts = correction.split(',')
                    if len(parts) != 3:
                        print(f"Invalid format for correction: {correction}")
                        continue
                    
                    row = int(parts[0]) - 1  # Convert to 0-based indexing
                    col = int(parts[1]) - 1  # Convert to 0-based indexing
                    value = int(parts[2])
                    
                    if row < 0 or row > 8 or col < 0 or col > 8 or value < 0 or value > 9:
                        print(f"Invalid values in correction: {correction} (row/col: 1-9, value: 0-9)")
                        continue
                    
                    # Apply the correction
                    old_value = self.current_board.grid[row, col]
                    self.current_board.grid[row, col] = value
                    print(f"Applied correction: ({row+1},{col+1}) {old_value} → {value}")  # Display with 1-based indexing
                    corrections_applied = True
                    
                except ValueError:
                    print(f"Invalid format for correction: {correction}")
            
            if corrections_applied:
                # Re-validate the board
                is_valid = self.current_board.validate()
                self.current_board.is_valid = is_valid
                
                print(f"\nBoard is now {'VALID ✅' if is_valid else 'INVALID ❌'}")
                
                if not is_valid:
                    self._identify_problematic_cells()
                    if self.problematic_positions:
                        print(f"Still have {len(self.problematic_positions)} problematic handwritten cells.")
                    else:
                        print("No specific problematic handwritten cells identified.")
                else:
                    print("Board is now valid! You can now get hints or see the solution.")
                    break
    
    def suggest_hint(self) -> Optional[HintSuggestion]:
        """Suggest a helpful next move."""
        if self.current_board is None or not self.current_board.is_valid:
            print("Cannot provide hints for invalid boards.")
            return None
            
        # Find possible values for each empty cell
        empty_cells = self.current_board.get_empty_cells()
        
        if not empty_cells:
            print("Board is already complete!")
            return None
        
        best_hint = None
        min_possibilities = 10
        
        for row, col in empty_cells:
            possibilities = []
            
            # Check which numbers are valid for this cell
            for num in range(1, 10):
                if self._is_valid_placement(row, col, num):
                    possibilities.append(num)
            
            # If only one possibility, this is a great hint
            if len(possibilities) == 1:
                best_hint = HintSuggestion(
                    row=row,
                    col=col,
                    value=possibilities[0],
                    reason=f"Only possible value for cell ({row+1}, {col+1})",  # 1-based indexing
                    difficulty="easy"
                )
                break
            elif len(possibilities) < min_possibilities and len(possibilities) > 0:
                min_possibilities = len(possibilities)
                best_hint = HintSuggestion(
                    row=row,
                    col=col,
                    value=possibilities[0],  # Just pick the first one
                    reason=f"Cell ({row+1}, {col+1}) has only {len(possibilities)} possibilities: {possibilities}",  # 1-based indexing
                    difficulty="medium" if len(possibilities) <= 3 else "hard"
                )
        
        if best_hint:
            self._show_hint(best_hint)
            self.last_hint = best_hint  # Store the last hint for potential application
            return best_hint
        else:
            print("Could not find a helpful hint.")
            self.last_hint = None
            return None
    
    def _is_valid_placement(self, row: int, col: int, num: int) -> bool:
        """Check if placing num at (row, col) would be valid."""
        # Temporarily place the number
        original = self.current_board.grid[row, col]
        self.current_board.grid[row, col] = num
        
        # Check validity
        valid = True
        
        # Check row
        row_values = self.current_board.grid[row, :]
        if len(row_values[row_values == num]) > 1:
            valid = False
        
        # Check column
        if valid:
            col_values = self.current_board.grid[:, col]
            if len(col_values[col_values == num]) > 1:
                valid = False
        
        # Check 3x3 box
        if valid:
            box_row, box_col = 3 * (row // 3), 3 * (col // 3)
            box = self.current_board.grid[box_row:box_row+3, box_col:box_col+3]
            if len(box[box == num]) > 1:
                valid = False
        
        # Restore original value
        self.current_board.grid[row, col] = original
        
        return valid
    
    def _show_hint(self, hint: HintSuggestion):
        """Display the hint visually."""
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        # Show the board
        self._show_digital_board(ax, title=f"Hint: Place {hint.value} at position ({hint.row+1}, {hint.col+1})")
        
        # Highlight the hint position
        ax.add_patch(plt.Rectangle((hint.col, hint.row), 1, 1, 
                                  fill=False, edgecolor='lime', linewidth=4))
        
        # Show the suggested number
        ax.text(hint.col + 0.5, hint.row + 0.5, str(hint.value),
               ha='center', va='center',
               color='lime',
               fontsize=20,
               fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n💡 HINT ({hint.difficulty.upper()}):")
        print(f"Place {hint.value} at row {hint.row+1}, column {hint.col+1}")  # 1-based indexing
        print(f"Reason: {hint.reason}")
        
        # Show option to apply hint
        print(f"\nOptions:")
        print(f"- Enter 'apply' to apply this hint and get the next one")
        print(f"- Continue with other menu options to do something else")
    
    def show_complete_solution(self):
        """Show the complete solution with color coding."""
        if self.current_board is None:
            print("No board loaded.")
            return
            
        if not self.current_board.is_valid:
            print("Cannot solve invalid board. Please fix validation issues first.")
            return
        
        # Make a copy of the board for solving
        solve_board = ImprovedSudokuBoard(self.current_board.grid.copy())
        solve_board.handwritten_mask = self.current_board.handwritten_mask.copy()
        
        # Solve the board
        print("Solving the puzzle...")
        solution_found = SudokuSolver.solve(solve_board)
        
        if not solution_found:
            print("Could not find a solution to this puzzle.")
            return
        
        self.solution = solve_board.solution
        
        # Show the solution with color coding
        plt.figure(figsize=(12, 10))
        ax = plt.gca()
        
        ax.fill([0, 9, 9, 0], [0, 0, 9, 9], 'white')

        # Draw grid lines
        for i in range(10):
            lw = 2 if i % 3 == 0 else 0.5
            ax.axhline(y=i, color='black', linewidth=lw)
            ax.axvline(x=i, color='black', linewidth=lw)

        # Add numbers with color coding
        for i in range(9):
            for j in range(9):
                value = self.solution[i, j]
                
                # Determine color based on origin
                if self.current_board.grid[i, j] != 0:
                    # Original number (printed or handwritten)
                    if self.current_board.handwritten_mask[i, j] == 1:
                        color = 'darkblue'  # Handwritten
                    else:
                        color = 'black'  # Printed
                else:
                    # Filled by solution
                    color = 'darkgreen'  # Changed to darker green
                
                ax.text(j + 0.5, i + 0.5, str(value),
                       ha='center', va='center',
                       color=color,
                       fontsize=15,
                       fontweight='bold')

        ax.set_xlim(0, 9)
        ax.set_ylim(9, 0)
        ax.set_title("Complete Solution")
        ax.axis('equal')
        ax.axis('off')
        
        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                      markersize=10, label='Original Printed'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                      markersize=10, label='Original Handwritten'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
                      markersize=10, label='Solution Filled')
        ]
        plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        plt.show()
        
        print("✅ Solution found and displayed!")
        
        # Show statistics
        original_filled = np.sum(self.current_board.grid != 0)
        solution_filled = np.sum(self.solution != 0)
        filled_by_solution = solution_filled - original_filled
        
        print(f"\nSolution Statistics:")
        print(f"- Original filled cells: {original_filled}")
        print(f"- Cells filled by solution: {filled_by_solution}")
        print(f"- Total cells: {solution_filled}")
    
    def apply_hint_and_next(self) -> bool:
        """Apply the last hint and show the next hint."""
        if self.last_hint is None:
            print("No hint available to apply. Get a hint first.")
            return False
        
        if not self.current_board.is_valid:
            print("Cannot apply hints to invalid boards.")
            return False
        
        # Apply the hint
        row, col, value = self.last_hint.row, self.last_hint.col, self.last_hint.value
        
        # Check if the cell is still empty
        if self.current_board.grid[row, col] != 0:
            print(f"Cell ({row+1}, {col+1}) is no longer empty. Cannot apply hint.")
            return False
        
        # Apply the hint
        self.current_board.grid[row, col] = value
        self.applied_hints.append((row, col))
        
        print(f"✅ Applied hint: Placed {value} at position ({row+1}, {col+1})")
        
        # Clear the last hint since it's been applied
        self.last_hint = None
        
        # Check if board is complete
        if self.current_board.is_complete():
            print("🎉 Congratulations! The puzzle is complete!")
            
            # Show the completed board
            plt.figure(figsize=(12, 10))
            ax = plt.gca()
            self._show_digital_board(ax, title="Completed Puzzle!")
            
            # Add legend with applied hints
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                          markersize=10, label='Original Printed'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                          markersize=10, label='Original Handwritten'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
                          markersize=10, label='Applied Hints')
            ]
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.show()
            return True
        
        # Get the next hint
        print("\n🔍 Getting next hint...")
        next_hint = self.suggest_hint()
        
        if next_hint is None:
            print("No more hints available. The puzzle might be nearly complete!")
            
            # Show current board state
            plt.figure(figsize=(12, 10))
            ax = plt.gca()
            self._show_digital_board(ax, title="Current Board State")
            
            # Add legend
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='black', 
                          markersize=10, label='Original Printed'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkblue', 
                          markersize=10, label='Original Handwritten'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
                          markersize=10, label='Applied Hints')
            ]
            plt.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1, 1))
            
            plt.tight_layout()
            plt.show()
        
        return True
    
    def interactive_session(self, image_path: str):
        """Run a complete interactive session for processing a Sudoku image."""
        print("🧩 INTERACTIVE SUDOKU SOLVER")
        print("="*50)
        
        # Process the image
        success = self.process_image(image_path, display_results=True)
        
        if not success:
            print("Failed to process the image. Please check the image path and try again.")
            return
        
        # Main interaction loop
        while True:
            print("\n" + "="*50)
            print("AVAILABLE OPTIONS:")
            
            if self.current_board.is_valid:
                print("1. Get a hint")
                if self.last_hint is not None:
                    print("2. Apply hint and show next")
                    print("3. Show complete solution")
                    print("4. Manual cell correction")
                else:
                    print("2. Show complete solution")
                    print("3. Manual cell correction")
            else:
                print("1. View problematic cells")
                print("2. Manual cell correction")
                print("3. Show current board status")
            
            print("8. Show detection results again")
            print("9. Exit")
            print("="*50)
            
            choice = input("Choose an option: ").strip()
            
            if choice == '1':
                if self.current_board.is_valid:
                    self.suggest_hint()
                else:
                    self.show_problematic_cells()
            elif choice == '2':
                if self.current_board.is_valid:
                    if self.last_hint is not None:
                        completed = self.apply_hint_and_next()
                        if completed and self.current_board.is_complete():
                            break  # Puzzle completed, end session
                    else:
                        self.show_complete_solution()
                else:
                    self.manual_cell_correction()
            elif choice == '3':
                if self.current_board.is_valid:
                    if self.last_hint is not None:
                        self.show_complete_solution()
                    else:
                        self.manual_cell_correction()
                else:
                    self._show_board_status()
            elif choice == '4':
                if self.current_board.is_valid and self.last_hint is not None:
                    self.manual_cell_correction()
                else:
                    self.show_detection_results()
            elif choice == '8':
                self.show_detection_results()
            elif choice == '9':
                print("Thank you for using the Interactive Sudoku Solver! 👋")
                break
            else:
                print("Invalid choice. Please try again.")
                
            # Check for special commands after hint
            if self.last_hint is not None and choice not in ['2']:
                user_input = input("\nEnter 'apply' to apply the last hint, or press Enter to continue: ").strip().lower()
                if user_input == 'apply':
                    completed = self.apply_hint_and_next()
                    if completed and self.current_board.is_complete():
                        break  # Puzzle completed, end session