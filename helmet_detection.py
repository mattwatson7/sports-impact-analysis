
import cv2
import torch
import numpy as np
import warnings
import csv
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial import distance as dist
from filterpy.kalman import KalmanFilter
from norfair import Detection, Tracker
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=FutureWarning)

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5x', force_reload=True)

# Reference object real-world size (e.g., player's height in meters)
reference_height_meters = 1.8  # Adjust this based on known object size in meters
# Dynamically estimate reference height in pixels for each frame
reference_height_pixels = None  # Dynamically estimate the reference height in each frame using detected players  # Actual measurement of the object in pixels

# Calculate real-world distance per pixel (meters per pixel)
# Calculate real-world distance per pixel using camera calibration if available
if reference_height_pixels is not None:
    real_world_distance_per_pixel = reference_height_meters / reference_height_pixels
else:
    # Fallback to using average calibration value or calibrate using chessboard pattern, etc.
    real_world_distance_per_pixel = 0.006  # Example default value, replace with calibrated data if available

# Define the mass of the players (e.g., estimated mass in kg)
# Define the mass of players dynamically based on estimated player size if possible
# Estimate player mass dynamically based on bounding box size or use the default value
player_mass = lambda bbox_size: max(60, min(120, 90 * (bbox_size / 30000)))  # Mass is scaled based on player size, capped between 60-120 kg  # Example mass in kg

# Open input video
video_path = "C:/Users/ilikp/Downloads/Tua_Concussion.mp4"
cap = cv2.VideoCapture(video_path)

# Check if the video was opened successfully
if not cap.isOpened():
    print(f"Error: Unable to open video file {video_path}")
    exit()

# Prepare output video
output_path = "C:/Users/ilikp/Downloads/ipro.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

# Get the first frame for optical flow
ret, first_frame = cap.read()

# Check if the first frame was successfully read
if not ret or first_frame is None:
    print("Error: Unable to read the first frame of the video.")
    cap.release()
    out.release()
    exit()

prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

# Previous velocities for players (by player ID)
previous_velocities = defaultdict(float)

# For time delta between frames (use actual frame rate)
fps = cap.get(cv2.CAP_PROP_FPS)
delta_time = 1 / fps

# Initialize Norfair tracker
tracker = Tracker(distance_function="euclidean", distance_threshold=100)

# Dictionary to map consistent IDs
player_positions = {}

# Threshold distance to consider it the same player (in pixels)
position_threshold = 100

# Unique ID counter for players
next_player_id = 1

# Kalman Filter initialization for each player
kalman_filters = {}

# Metrics dictionary for output CSV and graphing
metrics = defaultdict(list)

# Function to initialize a Kalman filter
def initialize_kalman_filter():
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.x = np.array([0, 0, 0, 0])  # Initial state (x, y, x_velocity, y_velocity)
    kf.F = np.array([[1, 0, delta_time, 0],
                     [0, 1, 0, delta_time],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])  # State transition matrix
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])  # Measurement function
    kf.P *= 1000  # Covariance matrix
    kf.R *= 5  # Measurement noise
    kf.Q = np.eye(4) * 0.1  # Process noise
    return kf

# Function to find the closest known player position
def get_consistent_id(center, player_positions):
    for player_id, last_position in player_positions.items():
        distance = np.linalg.norm(np.array(center) - np.array(last_position))
        if distance < position_threshold:
            return player_id
    return None

# Initialize Farneback optical flow parameters
farneback_params = dict(
    pyr_scale=0.5,
    levels=3,
    winsize=15,
    iterations=3,
    poly_n=5,
    poly_sigma=1.2,
    flags=0
)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert current frame to grayscale for optical flow
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calculate dense optical flow using Farneback's algorithm
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray_frame, None, **farneback_params)

    # Perform YOLOv5 inference for object detection
    results = model(frame)

    # Parse detections directly from results
    detections = []
    bounding_boxes = []
    for det in results.pred[0]:  # Access predictions from the results object
        x1, y1, x2, y2, conf, cls = det[:6]  # Extract bounding box and confidence
        if conf > 0.5 and int(cls) == 0:  # Filter out low-confidence detections and only consider people (class 0)
            width = x2 - x1
            height = y2 - y1
            side_length = max(width, height)  # Make bounding box a square
            x1 = int((x1 + x2) / 2 - side_length / 2)
            y1 = int((y1 + y2) / 2 - side_length / 2)
            x2 = int(x1 + side_length)
            y2 = int(y1 + side_length)
            bbox_area = (x2 - x1) * (y2 - y1)
            bounding_boxes.append([x1, y1, x2, y2])
            reference_height_pixels = bbox_area**0.5  # Update reference height dynamically based on the average bounding box
            # Norfair uses the center of the bounding box for tracking
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            detections.append(Detection(points=np.array([center_x, center_y])))

    # Update tracker with current detections
    tracked_objects = tracker.update(detections)

    # Calculate relative velocities between players
    player_velocities = {}  # Track velocity changes over time to estimate accelerations more accurately
    for idx, obj in enumerate(tracked_objects):
        center = obj.estimate[0]  # Get the estimated position from the tracker
        center_x, center_y = center[0], center[1]

        # Find a consistent ID or assign a new one
        consistent_id = get_consistent_id((center_x, center_y), player_positions)
        if consistent_id is None:
            consistent_id = next_player_id
            next_player_id += 1
            # Initialize Kalman filter with specific parameters like starting position
            kalman_filters[consistent_id] = initialize_kalman_filter()
            kalman_filters[consistent_id].x[:2] = np.array([center_x, center_y])  # Initialize with current position
            kalman_filters[consistent_id].P[:2, :2] = np.eye(2) * 100  # Increase initial uncertainty in position
            kalman_filters[consistent_id].x[:2] = np.array([center_x, center_y])  # Initialize with current position

        # Update player position
        player_positions[consistent_id] = (center_x, center_y)

        # Use Kalman filter to predict and update
        kalman = kalman_filters[consistent_id]
        kalman.predict()
        kalman.update([center_x, center_y])
        predicted_center = kalman.x[:2]

        # Calculate average flow magnitude inside the bounding box for velocity estimate
        if idx < len(bounding_boxes):
            x1, y1, x2, y2 = bounding_boxes[idx]
            flow_region = flow[max(0, int(y1)):min(int(y2), frame.shape[0]), max(0, int(x1)):min(int(x2), frame.shape[1])]
            if flow_region.size > 0:
                magnitude, _ = cv2.cartToPolar(flow_region[..., 0], flow_region[..., 1])
                # Calculate weighted average or filter outliers for better accuracy
                # Use weighted average or median for robustness to noise
                magnitude_filtered = magnitude[magnitude > np.percentile(magnitude, 10)]  # Remove low outliers
                average_magnitude = np.mean(magnitude_filtered)  # Using median to reduce noise
                if average_magnitude > 0:  # Update only if there is significant movement
                    current_velocity_pixels_per_sec = average_magnitude * fps
                    current_velocity_meters_per_sec = current_velocity_pixels_per_sec * real_world_distance_per_pixel

                    # Calculate change in velocity and force before updating previous velocity
                    delta_velocity = current_velocity_meters_per_sec - previous_velocities[consistent_id]
                    # Improved force calculation with damping factor
                    damping_coefficient = 0.05  # Introduce a variable damping coefficient
                    mass = player_mass(bbox_area)
                    force = mass * delta_velocity / delta_time - damping_coefficient * previous_velocities[consistent_id]

                    # Update previous velocity for the next frame
                    previous_velocities[consistent_id] = current_velocity_meters_per_sec
                    player_velocities[consistent_id] = current_velocity_meters_per_sec

                    # Draw bounding box around the person
                    label = f'Player {consistent_id}'
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                    # Display the velocity and force for the player
                    velocity_label = f'Velocity: {current_velocity_meters_per_sec:.2f} m/s'
                    force_label = f'Force: {force:.2f} N'
                    cv2.putText(frame, velocity_label, (int(x1), int(y1) - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                    cv2.putText(frame, force_label, (int(x1), int(y1) - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    # Highlight dangerous hits based on force threshold
                    dangerous_force_threshold = 1500  # Example threshold in Newtons
                    # Dynamically adjust dangerous force threshold based on player characteristics
                    mass = player_mass(bbox_area)
                    adjusted_threshold = dangerous_force_threshold * (mass / 90)  # Adjust based on estimated player mass
                    if abs(force) > adjusted_threshold:
                        cv2.putText(frame, 'Dangerous Hit!', (int(x1), int(y1) - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

                    # Store metrics for CSV and graph
                        metrics[consistent_id].append({
                        'frame': cap.get(cv2.CAP_PROP_POS_FRAMES),
                        'velocity': current_velocity_meters_per_sec,
                        'force': force
                    })

    # Write the processed frame to the output video
    out.write(frame)

    # Display the video with bounding boxes, velocity, force, and tracking info
    cv2.imshow('Force Estimation with Tracking', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Update previous frame for the next optical flow calculation
    prev_gray = gray_frame

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Write metrics to CSV file
csv_output_path = "C:/Users/ilikp/Downloads/player_metrics.csv"
with open(csv_output_path, mode='w', newline='') as csv_file:
    fieldnames = ['player_id', 'frame', 'velocity', 'force']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for player_id, player_metrics in metrics.items():
        for data in player_metrics:
            writer.writerow({'player_id': player_id, 'frame': data['frame'], 'velocity': data['velocity'], 'force': data['force']})

# Plot force and velocity over time for each player
for player_id, player_metrics in metrics.items():
    frames = [data['frame'] for data in player_metrics]
    velocities = [data['velocity'] for data in player_metrics]
    forces = [data['force'] for data in player_metrics]

    plt.figure()
    plt.plot(frames, velocities, label='Velocity (m/s)', color='b')
    plt.plot(frames, forces, label='Force (N)', color='r')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title(f'Player {player_id} Metrics Over Time')
    plt.legend()
    plt.savefig(f"C:/Users/ilikp/Downloads/player_{player_id}_metrics.png")
    plt.close()
