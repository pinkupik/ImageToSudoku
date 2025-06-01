# 🧩 Sudoku Detection and Solving System with web-based Streamlit interface

A comprehensive computer vision and AI-powered system that can detect, extract, and solve Sudoku puzzles from images. This project combines advanced image processing, OCR technology, and multiple solving algorithms to provide a complete Sudoku processing pipeline.

## ✨ Features

### 🔍 **Image Processing & Detection**
- **Robust Sudoku Board Detection**: Automatically detects Sudoku grids from photographs using advanced computer vision
- **Perspective Correction**: Handles angled or skewed images with automatic perspective transformation
- **Multiple Detection Methods**: Adapts to different image types (printed, handwritten, digital)
- **Grid Structure Analysis**: Intelligent 9x9 cell extraction with adaptive thresholding

### 🔤 **Optical Character Recognition (OCR)**
- **PaddleOCR Integration**: High-accuracy digit recognition optimized for Sudoku puzzles
- **Handwriting Support**: Specialized handling for handwritten digits (including blue pen)
- **Confidence Scoring**: Quality assessment for extracted digits
- **Multi-format Support**: Works with various image formats and qualities

### 🧠 **Advanced Solving Algorithms**
- **Basic Backtracking**: Traditional recursive solving approach
- **Advanced Constraint Propagation**: Intelligent solving with multiple optimization techniques:
  - Most Restricted Variable (MRV) heuristic
  - Naked Singles detection
  - Hidden Singles detection
  - Constraint satisfaction optimization
- **Performance Benchmarking**: Built-in tools to compare solver performance

### 🖥️ **User Interface**
- **Interactive Web App**: Clean Streamlit-based interface
- **Real-time Solving**: Instant puzzle solving with step-by-step visualization
- **Multiple Input Methods**: Upload images or input puzzles manually

## 🚀 Live Demo

Try the hosted version at: **[Crazy AI Sudoku Solver Ultra Edition](https://crazyaisudokusolverultraedition.streamlit.app/)**

> **Note**: The hosted version currently has limited image recognition capabilities due to PaddleX 3.0.0 compatibility issues in the cloud environment. Manual input works perfectly!

## 📦 Installation

Developed and tested on python 3.12.7 and 3.12.10. For best compatibility use one of these versions.
```bash
python --version
```

### Using UV (Recommended)

[UV](https://github.com/astral-sh/uv) is a fast Python package installer and resolver. It's significantly faster than pip and provides better dependency resolution.

#### 1. Install UV
```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

#### 2. Clone and Setup Project
```bash
# Clone the repository
git clone https://github.com/pinkupik/ImageToSudoku.git
cd motustom
git checkout semestral

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt #It will take longer than regular because of huge paddlepaddle library, which needs to be compiled
```

### Using Traditional Pip

```bash
# Clone the repository
git clone https://github.com/pinkupik/ImageToSudoku.git
cd motustom
git checkout semestral

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt #It will take longer than regular because of huge paddlepaddle library, which needs to be compiled
```

Make sure you have python3-dev and build-essential installed, or else installing pandas will throw errors
```bash
sudo apt install python3-dev
sudo apt install build-essential
```

## 🎯 Usage

### Web Application

Launch the interactive Streamlit interface:

```bash
streamlit run main.py
```

Then open your browser to `http://localhost:8501`

### Programmatic Usage

#### Advanced Solving
```python
from app.src import sudsolve

sudsolve.solve_advanced(board)
```

#### Image Detection
```python
from app.src.suddet import SudokuBoardDetector

detector = SudokuBoardDetector()
result = detector.detect_and_extract('path/to/sudoku_image.jpg')

if result['success']:
    extracted_board = result['board']
    # Process the extracted board
```

#### OCR Digit Recognition
```python
from app.src import sudscan

digits = sudscan.scan_table(board_image)
```

## 🧪 Testing

### Run All Tests
```bash
# Using pytest
pytest

# With verbose output
pytest -v

# Run specific test file
pytest app/tests/test_sudsolve_advanced.py
```

### Test Coverage
```bash
# Run tests with coverage
pytest --cov
```

## 📁 Project Structure

```
motustom/
├── main.py                         # Streamlit app entry point
├── requirements.txt                # Project dependencies
├── README.md                       # This file
│
├── app/                            # Main application package
│   ├── src/                        # Core algorithms
│   │   ├── suddet.py               # Sudoku board detection
│   │   ├── sudscan.py              # OCR digit extraction
│   │   └── sudsolve.py             # Advanced solver
│   │
│   ├── gui/                        # User interface components
│   │   ├── display.py              # Main Streamlit interface
│   │   ├── dinput.py               # Interactive puzzle input
│   │   ├── dsolved.py              # Puzzle output
│   │   ├── dempty.py               # Temporary puzzle
│   │   └── dtables.py              # Table shower handler
│   │
│   ├── tests/                      # Test suite
│   │   └── ...
│   │
│   └── utils/                      # Utility functions
│       └── sudcheck.py             # Utilities for working with sudoku
│
└── report.pdf                      # Quick talk about the project
```

## 🔧 Key Components

### Sudoku Detection (`suddet.py`)
- **SudokuBoardDetector**: Main detection class with multiple algorithms
- **Grid Detection**: Robust line detection and intersection finding
- **Perspective Correction**: Automatic image straightening
- **Cell Extraction**: Precise 9x9 grid cell isolation

### OCR Processing (`sudscan.py`)
- **OCRDigitExtractor**: PaddleOCR-based digit recognition
- **Preprocessing**: Image enhancement for better OCR accuracy
- **Confidence Filtering**: Quality-based digit validation
- **Batch Processing**: Efficient multi-cell recognition

### Solving Algorithms (`sudsolve.py`)
- **Basic Solver**: Traditional backtracking algorithm
- **Advanced Solver**: Constraint propagation with multiple heuristics
- **Performance Optimization**: State management and pruning
- **Validation**: Complete solution verification

### User Interface (`gui/`)
- **Interactive Input**: Visual 3x3 box grid for manual entry
- **Image Upload**: Drag-and-drop image processing
- **Real-time Solving**: Instant solution display
- **Error Handling**: User-friendly error messages

## 🚨 Known Issues

- **PaddleX 3.0.0 Compatibility**: Some cloud environments may have issues with the latest PaddleX version
- **Image Quality**: Very low-resolution or heavily distorted images may not detect properly
- **Handwriting Variation**: Extremely stylized handwriting may require preprocessing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🙏 Acknowledgments

- **PaddleOCR** for excellent OCR capabilities
- **OpenCV** for computer vision functions
- **Streamlit** for the web interface framework
- **NumPy** for numerical computing

## 📊 Performance

The advanced solver shows significant performance improvements over basic backtracking:

- **Easy Puzzles**: ~90% faster
- **Medium Puzzles**: ~75% faster  
- **Hard Puzzles**: ~60% faster
- **Logic-only Solutions**: Nearly instantaneous

---

**Happy Sudoku Solving! 🎉**
