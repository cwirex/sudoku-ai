# Sudoku AI Project

A Python package for Sudoku board detection, digit recognition, and puzzle generation.

## Overview

This project provides tools for:
- Detecting Sudoku boards in images
- Extracting and recognizing digits from Sudoku puzzles
- Generating synthetic Sudoku puzzles with various styles
- Training and evaluating models for digit recognition

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sudoku-ai.git
cd sudoku-ai
```

2. Create and activate a virtual environment:
```bash
python -m venv sudoku-env
source sudoku-env/bin/activate  # On Windows: sudoku-env\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
sudoku_project/
├── data/                    # For datasets
├── models/                  # Saved models
├── src/                     # Source code
│   ├── __init__.py
│   ├── board_detection.py   # Sudoku board detection
│   ├── generator.py         # Sudoku puzzle generation  
│   ├── data_processor.py    # Data processing utilities
│   ├── model_builder.py     # Model architecture definitions
│   └── utils.py             # Utility functions
├── notebooks/               # Jupyter notebooks for visualization
│   ├── board_detection.ipynb
│   ├── model_training.ipynb
│   └── dataset_generation.ipynb
├── requirements.txt         # Dependencies
└── README.md
```

## Usage

### Board Detection

```python
from src.board_detection import BoardExtractor

# Initialize the board extractor
extractor = BoardExtractor()

# Extract the board from an image
image = cv2.imread('path/to/image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
warped, cells = extractor.extract_board(image)

# Visualize the extracted board
import matplotlib.pyplot as plt
plt.imshow(warped)
plt.show()
```

### Model Training

See the `notebooks/02_model_training.ipynb` notebook for a complete example of:
- Loading and preparing Sudoku image data
- Training digit recognition and cell type classification models
- Evaluating and comparing model performance
- Saving and loading models for reuse

### Dataset Generation

```python
from src.generator import generate_sample_dataset

# Generate a small dataset of synthetic Sudoku puzzles
generator, samples = generate_sample_dataset(
    num_samples=10,
    output_dir='data/my_dataset'
)
```

## Notebooks

The project includes several Jupyter notebooks to demonstrate functionality:

1. `01_board_detection.ipynb`: Demonstrates board detection and cell extraction
2. `02_model_training.ipynb`: Shows how to train and evaluate models
3. `03_dataset_generation.ipynb`: Illustrates dataset generation for training

## Apple Silicon Optimization

This project is optimized for Apple Silicon (M1/M2/M4) chips, using:
- `tensorflow-macos` for Apple Silicon compatibility
- `tensorflow-metal` for GPU acceleration

## License

[MIT License](LICENSE)