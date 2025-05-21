"""
Utility functions for the Sudoku AI project.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

def setup_project_paths():
    """
    Setup and return project paths.
    
    Returns:
        Dict: Dictionary with project paths
    """
    # Get the current file directory
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Define project paths
    paths = {
        'project_root': current_dir,
        'data_dir': os.path.join(current_dir, 'data'),
        'models_dir': os.path.join(current_dir, 'models'),
        'notebooks_dir': os.path.join(current_dir, 'notebooks'),
    }
    
    # Create directories if they don't exist
    for path in paths.values():
        if not os.path.exists(path):
            os.makedirs(path)
            
    return paths

def save_model(model, model_name: str, model_dir: Optional[str] = None):
    """
    Save a model to disk.
    
    Args:
        model: The model to save
        model_name: Name of the model
        model_dir: Directory to save the model (defaults to project models dir)
    
    Returns:
        str: Path to the saved model
    """
    if model_dir is None:
        paths = setup_project_paths()
        model_dir = paths['models_dir']
    
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model_path

def load_model(model_name: str, model_dir: Optional[str] = None):
    """
    Load a model from disk.
    
    Args:
        model_name: Name of the model
        model_dir: Directory where the model is saved (defaults to project models dir)
    
    Returns:
        The loaded model
    """
    import tensorflow as tf
    
    if model_dir is None:
        paths = setup_project_paths()
        model_dir = paths['models_dir']
    
    model_path = os.path.join(model_dir, f"{model_name}.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model {model_path} not found.")
    
    return tf.keras.models.load_model(model_path)

def plot_grid(grid: np.ndarray, title: str = 'Sudoku Grid', figsize: tuple = (5, 5)):
    """
    Plot a Sudoku grid with values.
    
    Args:
        grid: Sudoku grid array (9x9)
        title: Plot title
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.fill([0, 9, 9, 0], [0, 0, 9, 9], 'white')

    # Draw grid lines
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        plt.axhline(y=i, color='black', linewidth=lw)
        plt.axvline(x=i, color='black', linewidth=lw)

    # Add numbers
    for i in range(9):
        for j in range(9):
            if grid[i, j] != 0:
                plt.text(j + 0.5, i + 0.5, str(grid[i, j]),
                       ha='center', va='center',
                       color='black',
                       fontsize=12,
                       fontweight='bold')

    plt.xlim(0, 9)
    plt.ylim(9, 0)
    plt.title(title)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def display_cell_grid(cells, figsize: tuple = (12, 12)):
    """
    Display all cells in a 9x9 grid.
    
    Args:
        cells: List of (cell_image, (row, col)) tuples
        figsize: Figure size
    """
    fig, axes = plt.subplots(9, 9, figsize=figsize)

    # Flatten cells into a 9x9 grid
    grid_cells = [[None for _ in range(9)] for _ in range(9)]
    for cell_img, (row, col) in cells:
        grid_cells[row][col] = cell_img

    for row in range(9):
        for col in range(9):
            axes[row, col].imshow(grid_cells[row][col])
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.suptitle("Extracted Cells (9x9 Grid)", fontsize=16, y=1.02)
    plt.show()

def visualize_predictions(model, data, class_names, title, num_samples=10, show_errors_only=False):
    """
    Visualize model predictions on sample images.
    
    Args:
        model: Trained model
        data: Tuple of (x_data, y_data)
        class_names: List of class names
        title: Plot title
        num_samples: Number of samples to visualize
        show_errors_only: Whether to show only incorrect predictions
    """
    import random
    
    x_data, y_data = data

    # Get predictions for all data
    y_pred = model.predict(x_data)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Get true labels for all data
    if len(y_data.shape) > 1:  # One-hot encoded
        y_true_classes = np.argmax(y_data, axis=1)
    else:  # Class indices
        y_true_classes = y_data

    # Find indices of incorrect predictions if show_errors_only is True
    if show_errors_only:
        incorrect_indices = np.where(y_pred_classes != y_true_classes)[0]
        if len(incorrect_indices) == 0:
            print(f"No errors found in predictions! Model predicted all {len(y_true_classes)} examples correctly.")
            return

        print(f"Found {len(incorrect_indices)} incorrect predictions out of {len(y_true_classes)} examples.")

        # Get a subset of errors to display
        if len(incorrect_indices) <= num_samples:
            indices = incorrect_indices
        else:
            indices = np.random.choice(incorrect_indices, size=num_samples, replace=False)
    else:
        # Get random sample indices if we're not focusing on errors
        if len(x_data) <= num_samples:
            indices = range(len(x_data))
        else:
            indices = random.sample(range(len(x_data)), num_samples)

    # Create plot
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        if i >= len(axes):
            break

        # Display image
        axes[i].imshow(x_data[idx].reshape(28, 28), cmap='gray')

        # Add labels
        true_class = y_true_classes[idx]
        pred_class = y_pred_classes[idx]

        correct = true_class == pred_class
        color = 'green' if correct else 'red'

        axes[i].set_title(f"True: {class_names[true_class]}\nPred: {class_names[pred_class]}",
                        color=color)
        axes[i].axis('off')

    subtitle = "Incorrect Predictions Only" if show_errors_only else "Random Sample Predictions"
    plt.suptitle(f"{title} - {subtitle}")
    plt.tight_layout()
    plt.show()