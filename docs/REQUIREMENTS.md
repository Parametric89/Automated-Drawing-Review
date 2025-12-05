# Project Dependencies - ML Panel Detection System

## Python Version
- **Python 3.10+** (Project tested on Python 3.10.18)

---

## Core Dependencies

### Computer Vision & Image Processing
```
opencv-python>=4.8.0        # cv2 - Image processing, visualization
opencv-contrib-python       # Additional OpenCV modules
Pillow>=10.0.0             # PIL - Image manipulation
```

### Machine Learning & Deep Learning
```
torch>=2.0.0               # PyTorch - Deep learning framework
torchvision>=0.15.0        # PyTorch vision utilities
ultralytics>=8.0.0         # YOLOv8 - Object detection framework
```

### PDF Processing
```
PyPDF2>=3.0.0              # PDF manipulation
PyMuPDF>=1.23.0            # fitz - Advanced PDF processing
pdf2image>=1.16.0          # Convert PDF to images
reportlab>=4.0.0           # PDF generation with vector graphics
```

### Data Processing & Analysis
```
numpy>=1.24.0              # Numerical computing
pandas>=2.0.0              # Data analysis and manipulation
```

### Visualization
```
matplotlib>=3.7.0          # Plotting and visualization
```

### Geometry & Spatial Operations
```
shapely>=2.0.0             # Geometric operations for polygons
```

### Configuration & Utilities
```
PyYAML>=6.0.0              # yaml - Configuration files
tqdm>=4.65.0               # Progress bars
openpyxl>=3.1.0            # Excel file support (for reports)
```

---

## Optional Dependencies (for specific features)

### Color Analysis
```
scikit-image>=0.21.0       # Advanced image processing
```

### Performance Optimization
```
numba>=0.57.0              # JIT compilation for speed
```

---

## Installation Commands

### Full Installation (Recommended)
```bash
pip install opencv-python opencv-contrib-python Pillow torch torchvision ultralytics PyPDF2 PyMuPDF pdf2image reportlab numpy pandas matplotlib shapely PyYAML tqdm openpyxl
```

### Minimal Installation (Core features only)
```bash
pip install opencv-python Pillow torch torchvision ultralytics numpy PyYAML tqdm
```

### PDF Processing Add-on
```bash
pip install PyPDF2 PyMuPDF pdf2image reportlab
```

---

## Conda Environment Setup (User's Environment: rhino_mcp)

### Create new environment:
```bash
conda create -n rhino_mcp python=3.10
conda activate rhino_mcp
```

### Install with conda (preferred for PyTorch):
```bash
# PyTorch with CUDA support
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Other packages
pip install opencv-python opencv-contrib-python Pillow ultralytics PyPDF2 PyMuPDF pdf2image reportlab numpy pandas matplotlib shapely PyYAML tqdm openpyxl
```

---

## System Requirements

### GPU Support (for training)
- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit:** 11.8 or 12.1
- **GPU Memory:** Minimum 6GB (RTX A3000 confirmed working)

### CPU Requirements
- **Minimum:** 8GB RAM
- **Recommended:** 16GB+ RAM for large image processing

### Disk Space
- **Minimum:** 10GB for models and datasets
- **Recommended:** 50GB+ for full training pipeline

---

## Verification Script

Run this to verify all dependencies are installed:
```python
import sys

packages = {
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'torch': 'torch',
    'torchvision': 'torchvision',
    'ultralytics': 'ultralytics',
    'PyPDF2': 'PyPDF2',
    'fitz': 'PyMuPDF',
    'pdf2image': 'pdf2image',
    'reportlab': 'reportlab',
    'numpy': 'numpy',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'shapely': 'shapely',
    'yaml': 'PyYAML',
    'tqdm': 'tqdm',
    'openpyxl': 'openpyxl'
}

print("Checking dependencies...\n")
missing = []

for module, package in packages.items():
    try:
        __import__(module)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - MISSING")
        missing.append(package)

if missing:
    print(f"\n❌ Missing {len(missing)} packages:")
    print(f"Install with: pip install {' '.join(missing)}")
else:
    print("\n✅ All dependencies installed!")
    
# Check PyTorch CUDA
try:
    import torch
    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except:
    pass
```

---

## Package Versions Used in Development

```
opencv-python==4.8.1.78
Pillow==10.0.1
torch==2.5.1+cu121
torchvision==0.20.1+cu121
ultralytics==8.0.196
PyPDF2==3.0.1
PyMuPDF==1.23.8
pdf2image==1.16.3
reportlab==4.0.7
numpy==1.24.3
pandas==2.1.1
matplotlib==3.8.0
shapely==2.0.2
PyYAML==6.0.1
tqdm==4.66.1
openpyxl==3.1.2
```

---

## Troubleshooting

### Common Issues

**1. OpenCV Import Error**
```bash
# If cv2 import fails, try:
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python opencv-contrib-python
```

**2. PyTorch CUDA Not Available**
```bash
# Reinstall PyTorch with CUDA support:
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**3. PyMuPDF (fitz) Import Error**
```bash
pip install --upgrade PyMuPDF
```

**4. Shapely Import Error on Windows**
```bash
# Install from conda-forge
conda install -c conda-forge shapely
```

---

**Last Updated:** 2025-11-10
**Environment:** Windows 10/11, Python 3.10, CUDA 12.1

