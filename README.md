# sports impact analysis
interprofessional project for video-based analysis of player motion and collision forces leveraging deep learning and optical flow

This project processes a video of players, detecting and tracking them using YOLOv5 and Norfair, estimating their velocities and forces based on optical flow and Kalman filtering, and outputs an annotated video along with CSV metrics and plots.

Features

Object Detection: Uses YOLOv5x to detect people (class 0).

Tracking: Norfair tracker with Euclidean distance and a Kalman Filter per player for smoothing.

Optical Flow: Dense Farneback optical flow to estimate pixel velocities.

Real-World Estimation: Converts pixel velocities to meters/second using a reference size calibration.

Force Calculation: Estimates force on each player by mass × acceleration with damping.

Dangerous Hit Highlighting: Annotates "Dangerous Hit!" on frames when force exceeds a threshold.

Metrics Output:

Annotated output video (ipro.mp4).

CSV file with per-player frame, velocity, and force metrics (player_metrics.csv).

PNG plots of velocity and force over time for each player.

Requirements

Python 3.8+

torch and ultralytics (for YOLOv5)

opencv-python

numpy

scipy

filterpy

norfair

scikit-learn

matplotlib

Install via:

pip install torch opencv-python numpy scipy filterpy norfair scikit-learn matplotlib

Configuration

Edit the following variables at the top of the script:

video_path: Path to the input video file.

output_path: Path where the annotated output video will be saved.

csv_output_path: Path where the CSV metrics file will be written.

real_world_distance_per_pixel: Calibrated meters-per-pixel value (default 0.006).

reference_height_meters: Known object height in meters (e.g., player height).

player_mass: Lambda/function to estimate mass (in kg) based on bounding box size.

dangerous_force_threshold: Base threshold in Newtons for flagging a dangerous hit.

Usage

python force_estimation.py

The script will:

Open the input video.

Initialize YOLOv5 model and Norfair tracker.

Read each frame, detect players, track them, and compute optical flow.

Estimate velocity (m/s) and force (N) per player.

Annotate frames with bounding boxes, IDs, velocities, forces, and warnings.

Write the annotated video to output_path.

Save per-player metrics to CSV.

Generate and save plots for each player in PNG format.

Output

Annotated Video: Shows bounding boxes, IDs, velocity, force, and dangerous hit flags.

CSV Metrics (player_metrics.csv):

player_id: Unique tracker ID

frame: Frame number

velocity: Velocity in m/s

force: Force in N

Plots: player_<ID>_metrics.png, showing velocity and force vs. frame number.

Customization

Adjust Detection Confidence: Change the conf > 0.5 filter.

Tracker Sensitivity: Tweak distance_threshold in Tracker(...).

Optical Flow Params: Modify farneback_params for flow quality vs. speed.

Kalman Filter: Tune process noise (Q), measurement noise (R), and initial covariance (P).

Troubleshooting

Video Not Opening: Verify video_path and file permissions.

No Detections: Check model download, internet connection, or raise conf threshold.

Mis-tracking: Adjust position_threshold, tracker params, or re-initialize Kalman filters.
