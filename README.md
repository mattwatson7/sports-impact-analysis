# Player Force & Velocity Estimation


Real-time detection, tracking, and force estimation for players in video using YOLOv5, Norfair, Farneback optical flow, and Kalman filtering.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output](#output)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)

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
git clone https://github.com/mattwatson7/player-force-velocity-estimation.git
cd player-force-velocity-estimation
pip install -r requirements.txt
```

Or install required packages directly:

```bash
pip install torch opencv-python numpy scipy filterpy norfair scikit-learn matplotlib
```

## Configuration

Edit the top of `force_estimation.py` to configure:

| Variable                     | Description                                                    |
| ---------------------------- | -------------------------------------------------------------- |
| `video_path`                 | Path to input video file.                                      |
| `output_path`                | Path to save annotated output video.                           |
| `csv_output_path`            | Path to write CSV metrics.                                     |
| `real_world_distance_per_pixel` | Meters-per-pixel calibration (default `0.006`).              |
| `reference_height_meters`    | Known object height in meters (e.g., 1.8 m).                  |
| `player_mass`                | Lambda to estimate mass from bbox area (kg).                  |
| `dangerous_force_threshold`  | Base Newton threshold for dangerous hit flagging.             |

## Usage

Run the script:

```bash
python force_estimation.py
```

The script will:
1. Load the video and YOLOv5 model.
2. Detect and track players frame by frame.
3. Use optical flow to estimate pixel-level motion.
4. Convert velocity to meters/second using calibration.
5. Estimate force using Newton's second law.
6. Flag dangerous collisions visually.
7. Save an annotated video and data for analysis.

## Output

- **Annotated Video:** `ipro.mp4` with bounding boxes, IDs, velocities, forces, and alerts.
- **CSV File:** `player_metrics.csv` containing `[player_id, frame, velocity, force]`.
- **Graphs:** PNG plots per player showing velocity and force over time.

## Customization

- **Detection Threshold:** Change `conf > 0.5` to adjust YOLO sensitivity.
- **Tracking Precision:** Tweak Norfair's `distance_threshold`.
- **Optical Flow:** Modify `farneback_params` for different flow results.
- **Kalman Filter:** Adjust `kf.Q`, `kf.R`, and initial `kf.P` for smoother estimates.

## Troubleshooting

- **Video Not Loading:** Ensure `video_path` is valid and video format is supported.
- **No Detections:** Try lowering the confidence threshold or testing a different model size.
- **Tracking Drift:** Adjust `position_threshold` or Kalman filter noise settings.
- **Unrealistic Values:** Re-calibrate your `real_world_distance_per_pixel`.

