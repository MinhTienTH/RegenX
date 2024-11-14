# Coffee Cherry Counter 🍒

An advanced computer vision application that detects and counts coffee cherries in different ripeness stages using Streamlit and OpenCV.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technical Details](#technical-details)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Requirements](#requirements)
- [Contributing](#contributing)

## Overview

The Coffee Cherry Counter is a sophisticated image processing tool that helps coffee farmers and agronomists analyze coffee cherry ripeness stages. Using computer vision techniques, it can detect and count cherries in three stages of ripeness: ripe (red), partially ripe (yellow), and unripe (green).

## Features

- 🎯 Accurate detection of coffee cherries using advanced HSV color ranges
- 🔍 Intelligent shape validation to distinguish cherries from other objects
- 📊 Texture analysis to differentiate between cherries and leaves
- 🎨 Color-coded visualization of detected cherries
- 📈 Real-time counting of cherries by ripeness stage
- 🖥️ User-friendly web interface built with Streamlit

## Technical Details

The system uses several advanced computer vision techniques:
- HSV color space analysis for robust color detection
- Gray-Level Co-occurrence Matrix (GLCM) for texture analysis
- Contour analysis for shape validation
- Morphological operations for noise reduction
- Sobel edge detection for feature enhancement

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/coffee-cherry-counter.git

# Navigate to the project directory
cd coffee-cherry-counter

# Install required packages
pip install -r requirements.txt
```

### Requirements
```txt
streamlit
opencv-python
numpy
Pillow
scipy
```

## Usage

1. Run the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload an image of coffee cherries using the file uploader

4. View the results:
   - Color-coded detection visualization
   - Count by ripeness stage
   - Detection confidence metrics

## How It Works

### 1. Image Processing Pipeline
```python
def detect_cherries(image):
    1. Convert image to HSV color space
    2. Apply color masks for different ripeness stages
    3. Remove noise using morphological operations
    4. Detect and validate cherry shapes
    5. Analyze texture features
    6. Count and classify cherries
```

### 2. Ripeness Classification
The system identifies three stages of ripeness:
- 🔴 Ripe (Red): HSV range [0-10, 120-255, 100-255]
- 🟡 Partially Ripe (Yellow): HSV range [20-35, 100-255, 100-255]
- 🟢 Unripe (Green): HSV range [35-85, 40-255, 40-255]

### 3. Shape Validation
Cherries are validated using multiple criteria:
- Circularity (0.5 - 1.2)
- Aspect ratio (0.7 - 1.4)
- Area constraints (50 - 2000 pixels)

### 4. Texture Analysis
Uses GLCM to calculate:
- Contrast
- Uniformity
- Edge density
