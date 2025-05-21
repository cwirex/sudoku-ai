# %% [markdown]
# # Sudoku Dataset Generation
# 
# This notebook demonstrates how to generate custom Sudoku puzzle datasets for training and testing.

# %%
# Import necessary packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image

# Add the src directory to the path so we can import our modules
sys.path.append('..')

# Import our modules
from src.config.generator_config import SudokuGeneratorConfig
from src.utils import setup_project_paths, plot_grid
from src.generation.puzzle_generator import SudokuPuzzleGenerator
from src.generation.image_generator import SudokuImageGenerator
from src.generation.dataset_generator import SudokuDatasetGenerator, generate_sample_dataset

# %%
# Setup paths
paths = setup_project_paths()
print("Project paths:")
for key, path in paths.items():
    print(f"- {key}: {path}")

# %% [markdown]
# ## Quick Dataset Generation
# 
# First, let's try the quick function to generate a small sample dataset.

# %%
# Generate a small sample dataset
output_dir = os.path.join(paths['data_dir'], 'quick_sample_dataset')
generator, samples = generate_sample_dataset(
    num_samples=10,
    output_dir=output_dir,
    display_samples=8
)

# %% [markdown]
# ## Custom Dataset Generation
# 
# Now let's create a more customized dataset with specific parameters.

# %%
# Create a custom configuration
config = SudokuGeneratorConfig(
    # Main configuration values
    cell_size=80,
    grid_padding=40,
    background_padding=120,
    mask_rates=[0.3, 0.5],  # Only use these mask rates (fewer empty cells)
    
    # Digit rendering configuration
    digit_config={
        "vertical_alignment_offset": 5,  # Fix for vertical alignment
        "text_position_jitter": (-1, 1)  # Less jitter for more consistent positioning
    }
)

# %%
# Create dataset generator with custom config
output_dir = os.path.join(paths['data_dir'], 'custom_dataset')
image_generator = SudokuImageGenerator(config)
dataset_generator = SudokuDatasetGenerator(image_generator, config, output_dir)

# %% [markdown]
# ### Generate a Single Puzzle
# 
# Let's first generate a single puzzle to verify our configuration.

# %%
# Generate a puzzle with the PuzzleGenerator
puzzle_generator = SudokuPuzzleGenerator(config)
grid, solution = puzzle_generator.generate(mask_rate=0.5)

# Display the grid
print("Generated Sudoku puzzle:")
plot_grid(grid, title="Sudoku Puzzle")

# Display the solution
print("\nSolution:")
plot_grid(solution, title="Sudoku Solution")

# %%
# Generate a single image with our custom configuration
image, info = image_generator.generate_image(
    grid=grid,
    mode="mixed",  # mixed mode: some printed, some handwritten
    difficulty="medium",
    background_style="paper_color"
)

# Display the image
plt.figure(figsize=(10, 10))
plt.imshow(image)
plt.title("Generated Sudoku Image (Mixed Mode)")
plt.axis('off')
plt.show()

# Print the generation info
print("Generation info:")
for key, value in info.items():
    if key == 'handwritten_positions':
        print(f"- {key}: {len(value)} positions")
    else:
        print(f"- {key}: {value}")

# %% [markdown]
# ### Generate a Dataset
# 
# Now let's generate a complete dataset with various combinations of parameters.

# %%
# Generate a dataset with custom parameters
samples = dataset_generator.generate_dataset(
    num_samples=20,
    modes=["printed", "mixed"],  # Only use printed and mixed modes
    difficulties=["easy", "medium"],  # Only use easy and medium difficulties
    background_styles=["paper_color", "unified"],  # Only use these background styles
    mask_rates=[0.3, 0.5],  # Only use these mask rates
    save_images=True,
    save_labels=True,
    display_samples=8,  # Display 8 samples from the generated dataset
    seed=42  # Set a random seed for reproducibility
)

# %% [markdown]
# ### Export the Dataset
# 
# Let's export our dataset as a zip archive so it can be easily shared or used for training.

# %%
# Export the dataset as a zip archive
archive_path = dataset_generator.export_as_archive()
print(f"Dataset archived at: {archive_path}")

# %% [markdown]
# ## Generate a Larger Dataset for Training
# 
# For model training, we often need larger datasets with more variability.

# %%
# Create a configuration for training data
training_config = SudokuGeneratorConfig(
    # Use default values for most parameters
    digit_config={
        "vertical_alignment_offset": 5  # Fix for vertical alignment
    }
)

# %%
# Create dataset generator for training data
training_output_dir = os.path.join(paths['data_dir'], 'training_dataset')
training_image_generator = SudokuImageGenerator(training_config)
training_dataset_generator = SudokuDatasetGenerator(training_image_generator, training_config, training_output_dir)

# %%
# Uncomment to generate a larger dataset for training
# Note: This will take a while to run!

# training_samples = training_dataset_generator.generate_dataset(
#     num_samples=300,  # 300 samples is a good starting point
#     modes=["printed", "mixed", "handwritten"],  # Use all modes
#     difficulties=["easy", "medium", "hard"],  # Use all difficulties
#     background_styles=["none", "paper_color", "unified"],  # Multiple background styles
#     save_images=True,
#     save_labels=True,
#     display_samples=5,  # Just show a few samples
#     seed=42
# )
# 
# # Export as a zip archive
# training_archive_path = training_dataset_generator.export_as_archive()
# print(f"Training dataset archived at: {training_archive_path}")

# %% [markdown]
# ## Custom Test Image Generation
# 
# Sometimes you might want to generate specific test images with known parameters.

# %%
# Generate a predetermined puzzle (less randomness)
# First, create a specific grid and solution

# Create a puzzle with 50% of cells empty
test_grid, test_solution = puzzle_generator.generate(mask_rate=0.5)

# Create a set of handwritten positions (for mixed mode)
non_empty_positions = [(i, j) for i in range(9) for j in range(9) if test_grid[i, j] != 0]
handwritten_count = len(non_empty_positions) // 2  # Make half of the filled cells handwritten
handwritten_positions = set(random.sample(non_empty_positions, handwritten_count))

# Generate the image with specific parameters
test_image, test_info = image_generator.generate_image(
    grid=test_grid,
    mode="mixed",
    difficulty="medium",
    background_style="unified",
    handwritten_positions=handwritten_positions
)

# Display the test image
plt.figure(figsize=(12, 12))
plt.imshow(test_image)
plt.title("Custom Test Image")
plt.axis('off')
plt.show()

# Save the test image for later use
test_image_path = os.path.join(paths['data_dir'], 'custom_test_image.png')
Image.fromarray(test_image).save(test_image_path)
print(f"Test image saved to: {test_image_path}")

# %% [markdown]
# ## Visualizing Handwritten Positions
# 
# Let's visualize the handwritten positions on the grid to better understand the mixed mode.

# %%
# Create a mask image showing handwritten positions
handwritten_mask = np.zeros((9, 9))
for pos in handwritten_positions:
    handwritten_mask[pos[0], pos[1]] = 1

# Create a figure with the grid and the handwritten mask
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Draw the grid
ax1.imshow(np.ones((9, 9)), cmap='Greys', alpha=0.5)
for i in range(10):
    lw = 2 if i % 3 == 0 else 0.5
    ax1.axhline(y=i, color='black', linewidth=lw)
    ax1.axvline(x=i, color='black', linewidth=lw)

# Add numbers to the grid
for i in range(9):
    for j in range(9):
        if test_grid[i, j] != 0:
            # Color differently based on whether it's handwritten or printed
            color = 'blue' if (i, j) in handwritten_positions else 'black'
            ax1.text(j + 0.5, i + 0.5, str(test_grid[i, j]),
                   ha='center', va='center',
                   color=color,
                   fontsize=12,
                   fontweight='bold')

ax1.set_title("Grid with Handwritten Digits in Blue")
ax1.axis('off')

# Draw the handwritten mask
ax2.imshow(handwritten_mask, cmap='Blues')
ax2.set_title("Handwritten Positions (Blue)")
for i in range(9):
    for j in range(9):
        ax2.text(j, i, '1' if handwritten_mask[i, j] == 1 else '0',
                ha='center', va='center',
                color='white' if handwritten_mask[i, j] == 1 else 'black',
                fontsize=8)
ax2.set_xticks(np.arange(9))
ax2.set_yticks(np.arange(9))
ax2.set_xticklabels(np.arange(9))
ax2.set_yticklabels(np.arange(9))
ax2.grid(True, color='gray', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've demonstrated how to:
# 1. Generate individual Sudoku puzzles
# 2. Create Sudoku puzzle images with various styles
# 3. Generate complete datasets for training and testing
# 4. Export datasets as zip archives
# 5. Create custom test images with specific parameters
# 
# These capabilities are essential for training and evaluating Sudoku recognition systems, allowing you to generate synthetic training data with full control over the puzzle properties.


