# %% [markdown]
# # Improved Sudoku Processing with Multiple Predictions
# 
# This notebook demonstrates an enhanced Sudoku processing system that handles uncertain digit predictions by considering multiple possible values for each cell. It can identify and automatically correct problematic cells that cause invalid boards.

# %%
# Import necessary packages
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random

# Add the src directory to the path so we can import our modules
sys.path.append('..')

# Import our modules
from src.utils import load_model, plot_grid, setup_project_paths
from src.processing.solver import SudokuProcessor
# Import the improved solver
from src.processing.improved_solver import ImprovedSudokuProcessor, ImprovedSudokuBoard

# %%
# Setup paths
paths = setup_project_paths()
print("Project paths:")
for key, path in paths.items():
    print(f"- {key}: {path}")

# %% [markdown]
# ## Initialize the Improved Sudoku Processor
# 
# The improved processor considers multiple predictions for each cell and can automatically correct problematic cells.

# %%
# Check for available models
def list_models():
    """List available models in the models directory."""
    models_dir = paths['models_dir']
    if not os.path.exists(models_dir):
        print(f"Models directory does not exist: {models_dir}")
        return []
        
    models = [os.path.splitext(f)[0] for f in os.listdir(models_dir) if f.endswith('.h5')]
    return models

models = list_models()
print(f"Available models: {models}")

# If no models are available, we need to train them first
if not models:
    print("No trained models found. Please run the model training notebook first.")

# %%
# Initialize the improved Sudoku processor
# It will capture the top-3 predictions for each cell by default
improved_processor = ImprovedSudokuProcessor(
    # Uncomment and set these if you know the specific model names
    # digit_model_path="deeper_cnn_digits",
    # cell_type_model_path="deeper_cell_classifier",
    confidence_threshold=0.5,  # Minimum confidence for primary prediction
    top_n=3  # Keep top-3 predictions for each cell
)

# %% [markdown]
# ## Load Test Images
# 
# We'll use both successful and problematic test images to demonstrate the improvements.

# %%
# Look for test images
from src.generation.image_generator import SudokuImageGenerator
from src.generation.puzzle_generator import SudokuPuzzleGenerator
from src.processing.data_processor import SudokuDataset
from src.config.generator_config import SudokuGeneratorConfig


# Try to find existing test images first
test_images = []
test_image_paths = []

# Check for synthetic image
synthetic_path = os.path.join(paths['data_dir'], 'synthetic_test.png')
if os.path.exists(synthetic_path):
    test_image_paths.append((synthetic_path, 'Synthetic Test'))

# Check other potential image paths
custom_path = os.path.join(paths['data_dir'], 'custom_test_image.png')
if os.path.exists(custom_path):
    test_image_paths.append((custom_path, 'Custom Test'))

# Look for any dataset sample
dataset_name = 'mini_sudoku_dataset'
dataset = SudokuDataset()

if os.path.exists(os.path.join(paths['data_dir'], f'{dataset_name}.zip')):
    # Load the dataset
    success = dataset.load_dataset(dataset_name)
    if success:
        # Load some samples
        samples = dataset.load_samples(max_samples=3)
        if samples:
            for i, sample in enumerate(samples):
                # Save the sample image to a temporary file
                temp_path = os.path.join(paths['data_dir'], f'temp_test_image_{i}.jpg')
                cv2.imwrite(temp_path, cv2.cvtColor(sample['image'], cv2.COLOR_RGB2BGR))
                test_image_paths.append((temp_path, f'Dataset Sample {i+1}'))

if not test_image_paths:
    print("No test images found. Generating synthetic ones...")
    # Generate synthetic images if no test images are found
    config = SudokuGeneratorConfig(
        digit_config={"vertical_alignment_offset": 5}
    )
    
    # Create generators
    puzzle_generator = SudokuPuzzleGenerator(config)
    image_generator = SudokuImageGenerator(config)
    
    # Generate two puzzles - one normal and one with noise to cause errors
    for i in range(2):
        # Generate a puzzle
        grid, solution = puzzle_generator.generate(mask_rate=0.5)
        
        # For the second image, add some noise to make recognition harder
        difficulty = "hard" if i == 1 else "medium"
        
        # Calculate handwritten positions for mixed mode
        non_empty_positions = [(r, c) for r in range(9) for c in range(9) if grid[r, c] != 0]
        handwritten_count = len(non_empty_positions) // 2
        handwritten_positions = set(random.sample(non_empty_positions, handwritten_count))
        
        # Generate image
        image, _ = image_generator.generate_image(
            grid=grid,
            mode="mixed",
            difficulty=difficulty,
            background_style="unified" if i == 0 else "texture",
            handwritten_positions=handwritten_positions
        )
        
        # Save the image
        path = os.path.join(paths['data_dir'], f'synthetic_test_{i}.png')
        cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        test_image_paths.append((path, f'Synthetic Test {i+1}'))
        
        # Store for reference
        test_images.append({
            'path': path,
            'name': f'Synthetic Test {i+1}',
            'grid': grid,
            'solution': solution,
            'difficulty': difficulty
        })

print(f"Found {len(test_image_paths)} test images:")
for path, name in test_image_paths:
    print(f"- {name}: {path}")

# %% [markdown]
# ## Process Images with Basic vs. Improved Processor
# 
# Let's compare the performance of the basic processor with our improved version.

# %%
# Initialize a regular processor for comparison
regular_processor = SudokuProcessor(confidence_threshold=0.5)

# %%
# Function to process an image with both processors and compare
def compare_processors(image_path, name):
    """Process an image with both regular and improved processors."""
    print(f"\n{'-'*50}\nProcessing {name}\n{'-'*50}")
    
    # Display the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"Test Image: {name}")
    plt.axis('off')
    plt.show()
    
    # Process with regular processor
    print("\nProcessing with regular processor...")
    reg_board = regular_processor.process_image(image_path, display_steps=False)
    
    if reg_board:
        # Display the extracted board
        reg_board.display(title=f"Regular Processor - {name}", show_cell_types=True)
        
        # Print board stats
        print("Regular processor stats:")
        reg_stats = reg_board.get_stats()
        for key, value in reg_stats.items():
            print(f"- {key}: {value}")
            
        # Try to solve
        if reg_board.is_valid:
            print("\nAttempting to solve with regular processor...")
            solution_found = regular_processor.solve_board(reg_board)
            if solution_found:
                print("Regular processor found a solution!")
                plot_grid(reg_board.solution, title=f"Regular Processor Solution - {name}")
            else:
                print("Regular processor could not find a solution.")
        else:
            print("\nBoard is invalid, regular processor will not attempt solving.")
    else:
        print("Regular processor failed to process the image.")
    
    # Process with improved processor (without auto-correction first)
    print("\nProcessing with improved processor...")
    imp_board = improved_processor.process_image(image_path, display_steps=False, auto_correct=False)
    
    if imp_board:
        # Display the extracted board with alternatives
        imp_board.display_with_alternatives(title=f"Improved Processor - {name} (Before Correction)")
        
        # Print board stats
        print("Improved processor stats (before correction):")
        imp_stats = imp_board.get_stats()
        for key, value in imp_stats.items():
            print(f"- {key}: {value}")
        
        # Print detailed cell predictions for problematic cells
        if not imp_board.is_valid:
            print("\nDetailed predictions for problematic cells:")
            improved_processor.print_cell_predictions(imp_board, min_confidence=0.1)
            
            # Try auto-correcting
            print("\nAttempting auto-correction...")
            corrected = imp_board.auto_correct_board()
            if corrected:
                print("Successfully corrected the board!")
                imp_board.display_with_alternatives(title=f"Improved Processor - {name} (After Correction)")
                
                # Try solving the corrected board
                print("\nAttempting to solve the corrected board...")
                solution_found = improved_processor.solve_board(imp_board)
                if solution_found:
                    print("Found a solution for the corrected board!")
                    plot_grid(imp_board.solution, title=f"Improved Processor Solution - {name}")
                else:
                    print("Could not find a solution for the corrected board.")
            else:
                print("Could not fully correct the board automatically.")
        else:
            # Board is already valid, try solving directly
            print("\nBoard is valid, attempting to solve...")
            solution_found = improved_processor.solve_board(imp_board)
            if solution_found:
                print("Improved processor found a solution!")
                plot_grid(imp_board.solution, title=f"Improved Processor Solution - {name}")
            else:
                print("Improved processor could not find a solution.")
    else:
        print("Improved processor failed to process the image.")
        
    return reg_board, imp_board

# %%
# Process each test image
results = []

for image_path, name in test_image_paths:
    reg_board, imp_board = compare_processors(image_path, name)
    results.append({
        'name': name,
        'path': image_path,
        'reg_board': reg_board,
        'imp_board': imp_board,
        'reg_valid': reg_board.is_valid if reg_board else False,
        'imp_valid': imp_board.is_valid if imp_board else False,
        'reg_solved': hasattr(reg_board, 'solution') and reg_board.solution is not None if reg_board else False,
        'imp_solved': hasattr(imp_board, 'solution') and imp_board.solution is not None if imp_board else False
    })

# %% [markdown]
# ## Manual Cell Override
# 
# If auto-correction doesn't work, we can manually override problematic cells.

# %%
def manually_correct_board(board):
    """Manually correct problematic cells in a board."""
    if not isinstance(board, ImprovedSudokuBoard):
        print("This function only works with ImprovedSudokuBoard objects.")
        return board
    
    # Make sure problematic cells are identified
    if not board.problematic_cells:
        board.validate_with_improvements()
    
    if not board.problematic_cells:
        print("No problematic cells identified.")
        return board
    
    print(f"Found {len(board.problematic_cells)} problematic cells:")
    for i, (row, col) in enumerate(board.problematic_cells):
        cell_pred = board.cell_predictions[row, col]
        if cell_pred:
            print(f"{i+1}. {cell_pred}")
    
    # Display the board with alternatives
    board.display_with_alternatives(title="Board with Problematic Cells")
    
    # Ask for manual corrections
    print("\nEnter manual corrections in the format 'row,col,value' (e.g., '2,3,5')")
    print("Enter multiple corrections separated by semicolons (e.g., '2,3,5; 4,5,9')")
    print("Enter 'done' when finished")
    
    while True:
        user_input = input("\nCorrections (or 'done'): ")
        if user_input.lower() == 'done':
            break
        
        # Parse the corrections
        corrections = [c.strip() for c in user_input.split(';')]
        for correction in corrections:
            try:
                parts = correction.split(',')
                if len(parts) != 3:
                    print(f"Invalid format for correction: {correction}")
                    continue
                
                row = int(parts[0])
                col = int(parts[1])
                value = int(parts[2])
                
                if row < 0 or row > 8 or col < 0 or col > 8 or value < 0 or value > 9:
                    print(f"Invalid values in correction: {correction}")
                    continue
                
                # Apply the correction
                print(f"Applying correction: ({row},{col}) = {value}")
                board.apply_correction(row, col, value)
                
            except ValueError:
                print(f"Invalid format for correction: {correction}")
        
        # Re-validate the board
        is_valid = board.validate_with_improvements()
        print(f"Board is now {'valid' if is_valid else 'invalid'}")
        
        # Display the updated board
        board.display_with_alternatives(title="Board After Corrections")
        
        if is_valid:
            print("Board is now valid! No more corrections needed.")
            break
        else:
            print(f"Still have {len(board.problematic_cells)} problematic cells:")
            for i, (row, col) in enumerate(board.problematic_cells):
                cell_pred = board.cell_predictions[row, col]
                if cell_pred:
                    print(f"{i+1}. {cell_pred}")
    
    return board

# %%
# Find an invalid board that couldn't be auto-corrected
for result in results:
    if not result['imp_valid'] and result['imp_board'] is not None:
        print(f"Found an invalid board from {result['name']} that needs manual correction:")
        
        # Get a fresh copy of the board (without prior auto-correction)
        board = improved_processor.process_image(result['path'], display_steps=False, auto_correct=False)
        
        if board:
            # Try manual correction
            corrected_board = manually_correct_board(board)
            
            # Attempt to solve the corrected board
            if corrected_board.is_valid:
                print("\nAttempting to solve the manually corrected board...")
                solution_found = improved_processor.solve_board(corrected_board)
                if solution_found:
                    print("Found a solution for the manually corrected board!")
                    plot_grid(corrected_board.solution, title=f"Solution After Manual Correction - {result['name']}")
                else:
                    print("Could not find a solution for the manually corrected board.")
            else:
                print("Board is still invalid after manual corrections.")
            
            break
else:
    print("No invalid boards found to demonstrate manual correction.")
    print("Try processing a more challenging image that produces an invalid board.")

# %% [markdown]
# ## Improvements Summary
# 
# The improved Sudoku processor offers several key advantages over the basic version:

# %%
# Print a summary of the results
print("SUMMARY OF RESULTS:\n")
print(f"{'-'*80}")
print(f"{'Image Name':<20} | {'Regular Valid':<15} | {'Improved Valid':<15} | {'Regular Solved':<15} | {'Improved Solved':<15}")
print(f"{'-'*80}")

for result in results:
    name = result['name']
    reg_valid = "Valid" if result['reg_valid'] else "Invalid"
    imp_valid = "Valid" if result['imp_valid'] else "Invalid"
    reg_solved = "Solved" if result['reg_solved'] else "Not Solved"
    imp_solved = "Solved" if result['imp_solved'] else "Not Solved"
    
    print(f"{name:<20} | {reg_valid:<15} | {imp_valid:<15} | {reg_solved:<15} | {imp_solved:<15}")

print(f"{'-'*80}")

# %% [markdown]
# ## Key Improvements
# 
# The improved Sudoku processor offers several advantages over the basic version:
# 
# 1. **Multiple Predictions**: Stores the top-N predictions for each cell instead of only the best guess
# 
# 2. **Conflict Detection**: Identifies specific cells causing validation failures
# 
# 3. **Auto-Correction**: Automatically tries alternative predictions to fix invalid boards
# 
# 4. **Manual Override**: Provides a user interface for manually correcting problematic cells
# 
# 5. **Detailed Visualization**: Shows alternative predictions and problematic cells clearly
# 
# These improvements significantly increase the robustness of the Sudoku processing pipeline, making it much more likely to successfully handle challenging puzzles with uncertain digit recognition.

# %% [markdown]
# ## Process Your Own Image
# 
# You can also process your own custom Sudoku image with the improved processor.

# %%
def process_custom_image_improved(image_path):
    """Process a custom Sudoku image with the improved processor."""
    # Check if the file exists
    if not os.path.exists(image_path):
        print(f"File not found: {image_path}")
        return
        
    # Display the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.title(f"Custom Image")
    plt.axis('off')
    plt.show()
    
    # Process with improved processor
    print("\nProcessing with improved processor...")
    board = improved_processor.process_image(image_path, display_steps=True, auto_correct=True)
    
    if board:
        # Display the extracted board with alternatives
        board.display_with_alternatives(title="Extracted Custom Board")
        
        # Print board stats
        print("Board Statistics:")
        stats = board.get_stats()
        for key, value in stats.items():
            print(f"- {key}: {value}")
        
        # Print cell predictions
        improved_processor.print_cell_predictions(board, min_confidence=0.2)
        
        if not board.is_valid:
            # Try manual correction if auto-correction didn't work
            print("\nBoard is invalid. Would you like to manually correct it? (y/n)")
            user_input = input()
            if user_input.lower() == 'y':
                board = manually_correct_board(board)
        
        # Try to solve the board if it's valid
        if board.is_valid:
            print("\nAttempting to solve the board...")
            solution_found = improved_processor.solve_board(board)
            if solution_found:
                print("Solution found!")
                plot_grid(board.solution, title="Solved Custom Puzzle")
            else:
                print("Could not find a solution.")
        else:
            print("\nBoard is still invalid. Cannot attempt to solve.")
    else:
        print("Failed to process the image.")
        
    return board

# %%
# Example usage with a custom image
# Replace with your own image path
custom_image_path = "../data/temp_test_image.jpg"
custom_board = process_custom_image_improved(custom_image_path)

# %% [markdown]
# ## Conclusion
# 
# The improved Sudoku processing system significantly enhances the robustness of the pipeline by:
# 
# 1. Storing multiple predictions for each cell instead of just the top prediction
# 2. Identifying specific cells that cause validation issues
# 3. Automatically trying alternative predictions to correct problematic cells
# 4. Providing a clear interface for manual correction when needed
# 
# This approach makes the system much more capable of handling challenging real-world Sudoku images where digit recognition might be uncertain. The key innovation is tracking alternative predictions and using them to fix validation issues, either automatically or with user assistance.


