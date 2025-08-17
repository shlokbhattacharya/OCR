# OCR Performance Comparison: Optimized vs Standard

A real-time handwritten digit recognition application that compares the performance between optimized and standard implementations of OCR processing. Built with Python, TensorFlow/Keras, OpenCV, and Tkinter.

## 🚀 Features

- **Real-time digit recognition** using CNN model trained on MNIST dataset
- **Performance comparison** between optimized and standard processing algorithms
- **Interactive drawing canvas** with pen/eraser tools
- **Multi-digit number detection** with intelligent grouping
- **Batch processing optimization** for improved inference speed
- **Live performance metrics** and detailed statistics
- **Visual annotations** showing detected digits and grouped numbers

## 📊 Performance Highlights

The optimized version includes several key improvements:

- **Batch processing**: Multiple digits processed simultaneously instead of one-by-one
- **Vectorized operations**: NumPy-based calculations for faster preprocessing
- **Optimized grouping**: Graph-based digit grouping with vectorized distance calculations
- **Memory efficiency**: Reduced memory allocations and copies

Typical performance improvements: **3-10x faster** prediction times compared to the standard implementation.

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- OpenCV
- PIL (Pillow)
- NumPy
- Tkinter (usually included with Python)

### Setup
```bash
# Clone the repository
git clone https://github.com/shlokbhattacharya/ocr_visualization.git
cd ocr_visualization

# Install dependencies
pip install -r requirements.txt
```

## 🎮 Usage

### Starting the Application
```bash
python src/OCR_main.py
```

### Using the Interface

1. **Draw digits** on the black canvas using your mouse
2. **Switch versions** using the "Switch to Standard/Optimized" button
3. **View performance** metrics in real-time
4. **Use tools**:
   - **Eraser**: Toggle between pen and eraser modes
   - **Undo**: Remove the last drawn stroke
   - **Clear**: Clear the entire canvas

### Performance Analysis
- **Current performance**: Live display of prediction times
- **Statistics panel**: Detailed performance metrics including:
  - Average, minimum, maximum prediction times
  - Percentile analysis (50th, 95th)
  - Performance trends over time
  - Direct comparison between versions

## 🔧 Key Components

### OCR Version Switcher (`ocr_version_switcher.py`)
- Main controller managing both implementations
- Performance measurement and statistics tracking  
- UI management and version switching
- Real-time performance comparison

### Base Implementation (`ocr_base.py`)
- Common drawing functionality
- Event handling for mouse interactions
- Basic image preprocessing utilities
- UI components (canvas, buttons, labels)

### Optimized Version (`ocr_optimized.py`)
- **Batch preprocessing**: Process multiple crops simultaneously
- **Vectorized operations**: NumPy-based calculations for bounding boxes
- **Memory-efficient**: Reduced allocations and efficient array operations

### Standard Version (`ocr_standard.py`)
- Traditional one-by-one processing
- Individual digit predictions
- Reference implementation for performance comparison

## ⚙️ Configuration

Edit `constants.py` to customize:

```python
class ProcessingConfig:
    THRESHOLD_VALUE = 50        # Image thresholding
    MIN_DIGIT_AREA = 30        # Minimum pixel area for digit detection
    STANDARD_STROKE_WIDTH = 5   # Drawing stroke width

class UIConfig:
    CANVAS_SIZE = 1024         # Canvas dimensions
    ERASER_MULTIPLIER = 5      # Eraser width multiplier
```

## 🧠 Algorithm Details

### Digit Detection Pipeline
1. **Bounding box detection**: Find connected components in the image
2. **Preprocessing**: Resize, center, and normalize each digit crop to 28x28
3. **CNN prediction**: Use trained MNIST model for digit classification
4. **Grouping**: Combine nearby digits into multi-digit numbers
5. **Visualization**: Draw bounding boxes and labels

### Optimization Techniques
- **Batch processing**: Stack multiple 28x28 arrays for single model call
- **Vectorized preprocessing**: NumPy operations instead of loops
- **Early returns**: Skip processing on empty or noise regions  
- **Efficient grouping**: Vectorized distance calculations for digit proximity
- **Memory reuse**: Minimize allocations in hot paths

## 📈 Performance Metrics

The application tracks several key metrics:

- **Prediction time**: End-to-end time from drawing to display
- **Processing efficiency**: Batch vs individual processing comparison
- **Memory usage**: Optimized vs standard memory allocation patterns
- **Accuracy**: Model prediction confidence and accuracy

## 📝 License

This project is open source. Please check the repository for specific license information.

## 🔮 Future Improvements

- [ ] Support for custom model architectures
- [ ] Additional optimization techniques (model quantization, TensorRT)
- [ ] Export functionality for processed images
- [ ] Batch drawing mode for performance testing
- [ ] Integration with other ML frameworks (PyTorch, ONNX)
- [ ] Mobile/web deployment options

## 🐛 Troubleshooting

### Common Issues

**Model not found error**:
```
Could not load model: [Errno 2] No such file or directory: 'src/model/CNN_model.keras'
```
Solution: Ensure your trained model file is placed at `src/model/CNN_model.keras`

**Import errors**:
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version compatibility (3.7+)

**Performance issues**:
- Try reducing canvas size in `constants.py`
- Ensure TensorFlow is using appropriate hardware acceleration
- Close other resource-intensive applications
---

