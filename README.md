# SuperPoint with RealSense Camera

This repository provides a Python implementation of the SuperPoint keypoint detector and descriptor, integrated with the Intel RealSense camera for real-time keypoint detection and matching. The implementation is optimized for CUDA-enabled GPUs and includes features such as:

- Real-time keypoint detection and descriptor extraction.
- Non-Maximum Suppression (NMS) for filtering keypoints.
- Two-way nearest neighbor matching for keypoint correspondences.
- Visualization of keypoints and matches on the RealSense camera feed.
- Configurable parameters for keypoint detection, matching, and visualization.

  ![SuperPoint Tracker_screenshot_17 03 2025](https://github.com/user-attachments/assets/2cce5546-f7f1-414b-9e62-233f31dd15b7)

https://github.com/user-attachments/assets/e9d36e4b-4375-4d3a-825e-e45968fc9e78


## Introduction to SuperPoint

SuperPoint is a state-of-the-art keypoint detection and descriptor extraction algorithm. It uses a deep convolutional neural network to simultaneously detect keypoints and compute their descriptors. The algorithm is designed to be robust to changes in lighting, scale, and viewpoint, making it suitable for real-world applications such as SLAM, structure-from-motion, and augmented reality.

<img width="544" alt="superpoint_architecture" src="https://github.com/user-attachments/assets/7d23d221-9fba-43a1-a268-4e915c320f68" />


### Key Features of SuperPoint

- **Joint Detection and Description**: SuperPoint detects keypoints and computes their descriptors in a single forward pass of the network.
- **Self-Supervised Training**: The model is trained on synthetic data and fine-tuned on real images using a self-supervised approach.
- **High Repeatability**: The detected keypoints are highly repeatable across different views of the same scene.
- **Efficient Inference**: The network is lightweight and can run in real-time on modern GPUs.

### SuperPoint Algorithm Overview

1. **Input**: A grayscale image of size H x W.
2. **Encoder**: The image is passed through a shared encoder to extract feature maps.
3. **Detector Head**: The feature maps are processed by the detector head to produce a heatmap of keypoint probabilities.
4. **Descriptor Head**: The feature maps are processed by the descriptor head to compute a dense descriptor map.
5. **Keypoint Extraction**: Keypoints are extracted from the heatmap using Non-Maximum Suppression (NMS).
6. **Descriptor Extraction**: Descriptors are sampled from the dense descriptor map at the locations of the detected keypoints.
7. **Matching**: Keypoints from different images are matched using a two-way nearest neighbor search.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [License](#license)

## Features

- **RealSense Integration**: Captures live frames from an Intel RealSense camera.
- **SuperPoint Network**: Uses a pre-trained SuperPoint model for keypoint detection and descriptor extraction.
- **CUDA Support**: Accelerates inference on NVIDIA GPUs.
- **Keypoint Matching**: Performs two-way nearest neighbor matching with configurable thresholds.
- **Visualization**: Displays keypoints, matches, and tracks on the camera feed.
- **Configurable Parameters**: Control keypoint detection, matching, and visualization settings via command-line arguments.

## Requirements

### Hardware
- **Intel RealSense Camera**: Compatible models include D415, D435, etc.
- **NVIDIA GPU**: Required for CUDA acceleration (optional but recommended).

### Software
- **Python 3.8+**
- **PyTorch**: For running the SuperPoint network.
- **OpenCV**: For image processing and visualization.
- **pyrealsense2**: For interfacing with the RealSense camera.
- **NumPy**: For numerical operations.

## Installation

### Clone the Repository:
```bash
git clone https://github.com/your-username/superpoint-realsense.git
cd superpoint-realsense

# SuperPoint with RealSense

## Install pyrealsense2

### If you have the RealSense SDK installed:
```bash
pip install pyrealsense2
```

## Usage

### Running the Script
To run the script with default settings:
```bash
python superpoint_realsense.py --cuda --weights_path models/superpoint_v1.pth --H 480 --W 640
```

### Command-Line Arguments

| Argument         | Description                                           | Default Value  |
|-----------------|-------------------------------------------------------|---------------|
| `--weights_path` | Path to the pre-trained SuperPoint weights file.     | `superpoint_v1.pth` |
| `--H`           | Height of the input image.                           | `480`         |
| `--W`           | Width of the input image.                            | `640`         |
| `--nms_dist`    | Non-Maximum Suppression (NMS) distance.              | `4`           |
| `--conf_thresh` | Confidence threshold for keypoint detection.         | `0.015`       |
| `--nn_thresh`   | Descriptor matching threshold.                       | `0.7`         |
| `--cuda`        | Enable CUDA acceleration (requires NVIDIA GPU).      | `False`       |
| `--no_display`  | Disable visualization (useful for headless systems). | `False`       |
| `--max_matches` | Maximum number of keypoint matches to keep.          | `None`        |

## Example Commands

### Run with CUDA and limit matches to 100:
```bash
python superpoint_realsense.py --cuda --weights_path models/superpoint_v1.pth --H 480 --W 640 --max_matches 100
```

### Run without CUDA and disable visualization:
```bash
python superpoint_realsense.py --weights_path models/superpoint_v1.pth --H 480 --W 640 --no_display
```

## Configuration

### Keypoint Detection
- Adjust `--conf_thresh` to control the sensitivity of keypoint detection.
- Use `--nms_dist` to set the distance for Non-Maximum Suppression.

### Keypoint Matching
- Set `--nn_thresh` to control the descriptor matching threshold.
- Use `--max_matches` to limit the number of matches.

### Visualization
- Use `--no_display` to disable the visualization window.
- Adjust `--H` and `--W` to change the resolution of the input image.

## Troubleshooting

### 1. RealSense Camera Not Detected
- Ensure the camera is properly connected and recognized by the system.
- Run the `realsense-viewer` tool to verify camera functionality.

### 2. CUDA Errors
- Ensure you have the correct version of PyTorch installed with CUDA support.
- Verify CUDA availability:
```python
import torch
print(torch.cuda.is_available())
```


