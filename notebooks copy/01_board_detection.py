# %% [markdown]
# # Sudoku Board Detection and Visualization
# 
# This notebook demonstrates how to use the Sudoku board detection module to extract boards from images and visualize the results.

# %%
# Import necessary packages
import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Add the src directory to the path so we can import our modules
sys.path.append('..')

# Import our modules
from src.board_detection import BoardExtractor, ProcessingParams
from src.data_processor import SudokuDataset
from src.utils import plot_grid, display_cell_grid

# %%
# Setup paths
from src.utils import setup_project_paths

paths = setup_project_paths()
print("Project paths:")
for key, path in paths.items():
    print(f"- {key}: {path}")

# %% [markdown]
# ## Load Dataset
# 
# We'll load a dataset containing Sudoku puzzle images. If you don't have one, you can generate one using the generator module.

# %%
# Load a dataset
dataset = SudokuDataset()

# If you have a dataset zip file
dataset_name = 'mini_sudoku_dataset'  # Change to your dataset name
if os.path.exists(os.path.join(paths['data_dir'], f'{dataset_name}.zip')):
    dataset.load_dataset(dataset_name)
else:
    # If you don't have a dataset, generate one
    from src.generator import generate_sample_dataset
    generator, samples = generate_sample_dataset(num_samples=10)
    dataset_name = os.path.basename(generator.output_dir)
    dataset.load_dataset(dataset_name)

# %%
# Print dataset info
dataset.print_dataset_info(detailed=True)

# %% [markdown]
# ## Load Samples and Visualize
# 
# Now we'll load some samples and visualize them.

# %%
# Load a few samples
samples = dataset.load_samples(max_samples=5)

# Display the first sample
plt.figure(figsize=(10, 10))
plt.imshow(samples[0]['image'])
plt.title(f"Sample {samples[0]['id']}\nMode: {samples[0]['mode']}, Difficulty: {samples[0]['difficulty']}")
plt.axis('off')
plt.show()

# Display the ground truth grid
plot_grid(samples[0]['grid'], title="Ground Truth Grid")

# %% [markdown]
# ## Board Extraction
# 
# Let's extract the board and cells from an image.

# %%
# Initialize the board extractor
extractor = BoardExtractor()

# Extract the board and cells from the first sample
sample = samples[0]
warped, cells = extractor.extract_board(sample['image'], display_steps=True)

# Display the extracted board
if warped is not None:
    plt.figure(figsize=(10, 10))
    plt.imshow(warped)
    plt.title("Extracted Board")
    plt.axis('off')
    plt.show()
else:
    print("Board extraction failed!")

# %% [markdown]
# ## Cell Extraction
# 
# Now let's visualize the extracted cells.

# %%
# Display the extracted cells if successful
if cells is not None:
    display_cell_grid(cells)
else:
    print("Cell extraction failed!")

# %% [markdown]
# ## Process Multiple Images
# 
# Let's process multiple images and evaluate success rate.

# %%
# Process all samples
results = []
for sample in samples:
    warped, cells = extractor.extract_board(sample['image'], display_steps=False)
    results.append((warped is not None, cells is not None))

# Calculate success rate
board_success = sum(1 for board, _ in results if board) / len(results)
cell_success = sum(1 for _, cells in results if cells) / len(results)

print(f"Board extraction success rate: {board_success:.2%} ({sum(1 for board, _ in results if board)}/{len(results)})")
print(f"Cell extraction success rate: {cell_success:.2%} ({sum(1 for _, cells in results if cells)}/{len(results)})")

# %% [markdown]
# ## Try a Custom Image
# 
# You can also try the extraction on a custom image.

# %%
def process_custom_image(image_path):
    """Process a custom image and display the results."""
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return
    
    # Convert to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Display the original image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.title("Custom Image")
    plt.axis('off')
    plt.show()
    
    # Extract the board and cells
    warped, cells = extractor.extract_board(image, display_steps=True)
    
    # Display the extracted board
    if warped is not None:
        plt.figure(figsize=(10, 10))
        plt.imshow(warped)
        plt.title("Extracted Board")
        plt.axis('off')
        plt.show()
    else:
        print("Board extraction failed!")
    
    # Display the extracted cells
    if cells is not None:
        display_cell_grid(cells)
    else:
        print("Cell extraction failed!")

# %%
# Example: Using a custom image 
# custom_image_path = "/path/to/your/custom/sudoku/image.jpg"
# process_custom_image(custom_image_path)

# For now, let's use one of our samples again
# Save the image to disk first
custom_sample = samples[-1]
custom_path = os.path.join(paths['data_dir'], 'temp_custom.jpg')
cv2.imwrite(custom_path, cv2.cvtColor(custom_sample['image'], cv2.COLOR_RGB2BGR))

# Process it as a "custom" image
process_custom_image(custom_path)

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've demonstrated how to extract Sudoku boards and cells from images using our `BoardExtractor` class. This is the first step in our Sudoku AI pipeline - after extracting the cells, we would typically run them through our digit recognition model to solve the puzzle.


