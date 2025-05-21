# %% [markdown]
# # Sudoku Model Training and Evaluation
# 
# This notebook demonstrates how to train, evaluate, and use the Sudoku digit recognition and cell type classification models.

# %%
# Import necessary packages
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Check for Apple Silicon and configure TensorFlow accordingly
import platform
if platform.processor() == 'arm':
    print("Running on Apple Silicon - enabling Metal acceleration")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        print("Metal acceleration enabled")
    else:
        print("No Metal GPU found")

# Add the src directory to the path so we can import our modules
sys.path.append('..')

# Import our modules
from processing.data_processor import SudokuDataset, SudokuDataProcessor
from src.model_builder import SudokuModels, ModelEvaluator, SudokuExperiment
from src.utils import setup_project_paths, visualize_predictions

# %%
# Setup paths
paths = setup_project_paths()
print("Project paths:")
for key, path in paths.items():
    print(f"- {key}: {path}")

# %% [markdown]
# ## Load Dataset and Extract Cells
# 
# First, we'll load a dataset and extract the cells for training.

# %%
# Set up the experiment (this handles the full pipeline)
experiment = SudokuExperiment()

# Check if we have a dataset available
dataset_name = '330_sudoku_dataset'  # Change to your dataset name
if os.path.exists(os.path.join(paths['data_dir'], f'{dataset_name}.zip')):
    # Load the dataset
    success = experiment.setup_dataset(dataset_name)
    if not success:
        print("Dataset loading failed. Generating a new one...")
        from src.generator import generate_sample_dataset
        generator, samples = generate_sample_dataset(num_samples=100)  # More samples for training
        dataset_name = os.path.basename(generator.output_dir)
        success = experiment.setup_dataset(dataset_name)
else:
    # Generate a dataset if we don't have one
    print("Dataset not found. Generating a new one...")
    from src.generator import generate_sample_dataset
    generator, samples = generate_sample_dataset(num_samples=100)  # More samples for training
    dataset_name = os.path.basename(generator.output_dir)
    success = experiment.setup_dataset(dataset_name)

# %%
# Load samples from the dataset
samples = experiment.load_samples(max_samples=None, difficulties=['easy', 'medium', 'hard'])  # Load all available samples
print(f"Loaded {len(samples)} samples")

# %%
# Extract cells and prepare training data
datasets = experiment.prepare_data()

# Visualize the dataset
experiment.visualize_data()

# %% [markdown]
# ## Train Models
# 
# Now let's train both digit recognition and cell type classification models.

# %%
# Build models
model_builder = experiment.build_models()

# %%
# Train digit recognition models
digit_models = experiment.train_digit_models(epochs=10)

# %%
# Train cell type classification models
cell_type_models = experiment.train_cell_type_models(epochs=10)

# %% [markdown]
# ## Evaluate Models
# 
# Let's evaluate our trained models and compare their performance.

# %%
# Evaluate digit recognition models
digit_comparison = experiment.evaluate_digit_models()

# %%
# Evaluate cell type classification models
cell_comparison = experiment.evaluate_cell_type_models()

# %% [markdown]
# ## Visualize Predictions
# 
# Let's look at some of the model predictions.

# %%
# Get the best digit recognition model (based on previous evaluation)
if 'digit_models' in experiment.results:
    # Find the best model name from evaluation results
    # Only include models with "digits" in the name
    digit_models_only = [name for name in digit_comparison['Model'] if 'digits' in name]
    best_model_name = digit_models_only[0]  # Get the first (best) digit model
    best_digit_model = experiment.model_evaluator.models[best_model_name]
    
    # Get test data
    digit_test_data = experiment.datasets['digit_recognition']['test']

    # After loading the best model and before visualizing
    X_test, y_test = experiment.datasets['digit_recognition']['test']
    y_pred = best_digit_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # Visualize random predictions
    visualize_predictions(
        best_digit_model,
        digit_test_data,
        class_names=[str(i) for i in range(10)],
        title=f"Digit Recognition - {best_model_name}",
        num_samples=10
    )
    
    # Visualize incorrect predictions
    visualize_predictions(
        best_digit_model,
        digit_test_data,
        class_names=[str(i) for i in range(10)],
        title=f"Digit Recognition - {best_model_name}",
        num_samples=10,
        show_errors_only=True
    )

# %%
# Get the best cell type classification model
if 'cell_type_models' in experiment.results:
    # Find the best model name from evaluation results
    best_cell_model_name = cell_comparison.iloc[0]['Model']
    best_cell_model = experiment.model_evaluator.models[best_cell_model_name]
    
    # Get test data
    cell_test_data = experiment.datasets['cell_type_classification']['test']
    
    # Visualize random predictions
    visualize_predictions(
        best_cell_model,
        cell_test_data,
        class_names=['Empty', 'Printed', 'Handwritten'],
        title=f"Cell Type Classification - {best_cell_model_name}",
        num_samples=10
    )
    
    # Visualize incorrect predictions
    visualize_predictions(
        best_cell_model,
        cell_test_data,
        class_names=['Empty', 'Printed', 'Handwritten'],
        title=f"Cell Type Classification - {best_cell_model_name}",
        num_samples=10,
        show_errors_only=True
    )

# %% [markdown]
# ## Save Models
# 
# Let's save our best models for future use.

# %%
from src.utils import save_model

# Save the best digit recognition model
if 'digit_models' in experiment.results:
    digit_model_path = save_model(best_digit_model, best_model_name)
    print(f"Digit model saved to: {digit_model_path}")
    
    # Save the best cell type classification model
    cell_model_path = save_model(best_cell_model, best_cell_model_name)
    print(f"Cell type model saved to: {cell_model_path}")

# %% [markdown]
# ## Load and Test Saved Models
# 
# Let's verify that we can load the saved models and use them for inference.

# %%
from src.utils import load_model

# Load the saved models
loaded_digit_model = load_model(best_model_name)
loaded_cell_model = load_model(best_cell_model_name)

# Verify that the loaded models work
# Get a single test sample
x_test, y_test = digit_test_data
x_sample = x_test[0:1]  # Get just one sample

# Make predictions with both original and loaded models
original_pred = np.argmax(best_digit_model.predict(x_sample), axis=1)[0]
loaded_pred = np.argmax(loaded_digit_model.predict(x_sample), axis=1)[0]

# Display the results
plt.figure(figsize=(5, 5))
plt.imshow(x_sample[0, :, :, 0], cmap='gray')
plt.title(f"Original model: {original_pred}, Loaded model: {loaded_pred}")
plt.axis('off')
plt.show()

print(f"Original model prediction: {original_pred}")
print(f"Loaded model prediction: {loaded_pred}")
print(f"Predictions match: {original_pred == loaded_pred}")

# %% [markdown]
# ## Conclusion
# 
# In this notebook, we've demonstrated the process of:
# 1. Loading and preparing Sudoku image data
# 2. Training digit recognition and cell type classification models
# 3. Evaluating and comparing model performance
# 4. Saving and loading models for reuse
# 
# These models can now be incorporated into a complete Sudoku solver pipeline that takes an image of a Sudoku puzzle, detects the board, extracts the cells, recognizes the digits, and solves the puzzle.


