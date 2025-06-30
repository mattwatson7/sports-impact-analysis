# sports impact analysis
interprofessional project for video-based analysis of player motion and collision forces leveraging deep learning and optical flow

This project processes a video of players, detecting and tracking them using YOLOv5 and Norfair, estimating their velocities and forces based on optical flow and Kalman filtering, and outputs an annotated video along with CSV metrics and plots.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Object Detection:** YOLOv5x detects people (class 0).  
- **Tracking:** Norfair tracker with Euclidean distance and per-player Kalman Filters.  
- **Optical Flow:** Dense Farneback algorithm estimates pixel velocities.  
- **Real-World Conversion:** Transforms pixel velocities to m/s via calibrated scale.  
- **Force Calculation:** Computes force (N) using mass × acceleration with damping.  
- **Danger Highlighting:** Flags "Dangerous Hit!" when force exceeds a threshold.  
- **Metrics Export:** Generates annotated video, CSV logs, and PNG plots.  

## Requirements

- Python 3.8+  
- torch & ultralytics  
- opencv-python  
- numpy  
- scipy  
- filterpy  
- norfair  
- scikit-learn  
- matplotlib  

## Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/<your-username>/player-force-velocity-estimation.git
cd player-force-velocity-estimation
pip install -r requirements.txt
