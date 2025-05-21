# 🧩 Sudoku AI: Interactive Computer Vision Solver

An advanced AI-powered Sudoku solver that uses computer vision to detect puzzles from images and provides intelligent solving assistance through an interactive web interface.

![Sudoku AI Demo](demo.gif)

## ✨ Features

- 🤖 **AI-powered digit recognition** (printed & handwritten)
- 🎯 **Smart board detection** and perspective correction
- 💡 **Intelligent hints** with difficulty levels
- ✏️ **Manual correction** for any errors
- 🎨 **Color-coded visualization**
- 🧩 **Step-by-step solving** guidance
- 🌐 **Interactive web interface** built with Streamlit
- 📊 **Real-time progress tracking**
- 🔧 **Auto-correction** for recognition errors

## 🚀 Quick Start

### Web Application (Recommended)

1. **Clone and Setup:**
```bash
git clone https://github.com/yourusername/sudoku-ai.git
cd sudoku-ai
python -m venv sudoku-env
source sudoku-env/bin/activate  # On Windows: sudoku-env\Scripts\activate
pip install -r requirements.txt
```

2. **Train Models (if not available):**
```bash
python notebooks/03_model_training.py
```

3. **Launch Web App:**
```bash
streamlit run sudoku_app.py
```

4. **Open in Browser:**
Navigate to `http://localhost:8501` and start solving!

### Using the Web Interface

1. **📁 Upload** a clear image of a Sudoku puzzle
2. **🔍 Process** the image to detect and extract the board
3. **💡 Get hints** for step-by-step solving guidance
4. **✏️ Fix errors** manually if needed
5. **🎯 View solution** or solve it completely

## 🎯 How It Works

### 1. Computer Vision Pipeline
- **Board Detection**: Automatically finds and extracts Sudoku grids from photos
- **Perspective Correction**: Straightens tilted or angled puzzle images
- **Cell Extraction**: Isolates individual cells for digit recognition

### 2. AI Recognition
- **Printed Digits**: Recognizes standard printed numbers
- **Handwritten Digits**: Handles various handwriting styles
- **Multiple Predictions**: Keeps backup predictions for error correction
- **Confidence Scoring**: Tracks prediction reliability

### 3. Intelligent Solving
- **Smart Hints**: Provides context-aware solving suggestions
- **Difficulty Levels**: Adapts hints based on puzzle complexity
- **Auto-Correction**: Fixes recognition errors automatically
- **Manual Override**: Allows user corrections when needed

## 📁 Project Structure

```
sudoku-ai/
├── 🌐 sudoku_app.py                 # Interactive Streamlit web app
├── 📂 src/                          # Core Python package
│   ├── 📄 __init__.py
│   ├── 🔍 board_detection.py        # Computer vision for board detection
│   ├── 🎲 generator.py              # Synthetic dataset generation
│   ├── 📊 data_processor.py         # Data loading and preprocessing
│   ├── 🧠 model_builder.py          # Neural network architectures
│   ├── 🎯 sudoku_solver.py          # Basic solving functionality
│   ├── ⚡ improved_solver.py        # Advanced solver with multi-predictions
│   ├── 🎮 interactive_solver.py     # Web app backend integration
│   └── 🛠️ utils.py                  # Utility functions
├── 📁 data/                         # Datasets and images
├── 🧠 models/                       # Saved neural network models
├── 📓 notebooks/                    # Demonstration notebooks
│   ├── 01_board_detection.py       # Board detection examples
│   ├── 02_dataset_generation.py    # Dataset creation demos
│   ├── 03_model_training.py        # Model training pipeline
│   └── 04_sudoku_solver.py         # Advanced solving examples
├── 📋 requirements.txt              # Python dependencies
└── 📖 README.md                     # This file
```

## 🛠️ Development Usage

### Board Detection
```python
from src.board_detection import BoardExtractor
import cv2

# Initialize the board extractor
extractor = BoardExtractor()

# Load and process an image
image = cv2.imread('sudoku_photo.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Extract the board and cells
warped_board, cells = extractor.extract_board(image, display_steps=True)
```

### Interactive Solving
```python
from src.interactive_solver import InteractiveSudokuSolver

# Initialize the interactive solver
solver = InteractiveSudokuSolver()

# Process an image
success = solver.process_image('sudoku_photo.jpg')

if success:
    # Get intelligent hints
    hint = solver.suggest_hint()
    print(f"Hint: Place {hint.value} at ({hint.row+1}, {hint.col+1})")
    
    # Apply hint and get next suggestion
    solver.apply_hint_and_next()
```

### Dataset Generation
```python
from src.generator import generate_sample_dataset

# Generate synthetic training data
generator, samples = generate_sample_dataset(
    num_samples=100,
    output_dir='data/training_set',
    modes=['printed', 'handwritten', 'mixed'],
    difficulties=['easy', 'medium', 'hard']
)
```

### Model Training
```python
from src.model_builder import SudokuExperiment

# Set up complete training pipeline
experiment = SudokuExperiment()
experiment.setup_dataset('my_sudoku_dataset')
experiment.prepare_data()

# Train models
experiment.build_models()
digit_models = experiment.train_digit_models(epochs=20)
cell_models = experiment.train_cell_type_models(epochs=15)

# Evaluate performance
experiment.evaluate_digit_models()
experiment.evaluate_cell_type_models()
```

## 📊 Model Performance

Our trained models achieve:
<!-- - **Digit Recognition**: >95% accuracy on printed digits, >90% on handwritten
- **Cell Classification**: >98% accuracy (empty/printed/handwritten)
- **Board Detection**: >95% success rate on clear images
- **End-to-End Solving**: >85% automatic solving success rate -->

## 🎨 Web Interface Screenshots

### Landing Page
![Landing Page](screenshots/landing.png)

### Puzzle Processing
![Processing](screenshots/processing.png)

### Interactive Solving
![Solving](screenshots/solving.png)

### Hint System
![Hints](screenshots/hints.png)

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM
- Camera or image files of Sudoku puzzles

### Key Dependencies
- **Streamlit**: Web interface framework
- **OpenCV**: Computer vision operations
- **TensorFlow**: Deep learning models
- **NumPy/Matplotlib**: Data processing and visualization
- **scikit-learn**: Model evaluation utilities

### Apple Silicon Optimization
Optimized for Apple Silicon (M1/M2/M3/M4) chips with:
- `tensorflow-macos` for native compatibility
- `tensorflow-metal` for GPU acceleration

## 🎯 Tips for Best Results

**✅ Good Images:**
- Clear, well-lit photos
- Straight-on angle
- High contrast between digits and background
- Minimal shadows or glare

**⚠️ Avoid:**
- Blurry or heavily tilted images
- Poor lighting conditions
- Reflections on the puzzle surface
- Partially visible or cut-off grids

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- OpenCV community for computer vision tools
- Streamlit team for the amazing web framework
- MNIST dataset for handwritten digit training data

## 📞 Support

If you encounter any issues or have questions:
- Open an issue on GitHub
- Check the notebooks for detailed examples
- Review the code documentation

---

<div align="center">

**Made with ❤️ and lots of ☕**

[⭐ Star this repo](https://github.com/yourusername/sudoku-ai) | [🐛 Report Bug](https://github.com/yourusername/sudoku-ai/issues) | [💡 Request Feature](https://github.com/yourusername/sudoku-ai/issues)

</div>