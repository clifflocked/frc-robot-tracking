import os

import numpy as np
import pandas as pd
import supervision as sv
from dotenv import load_dotenv

from time import time
from math import floor
import sys
from contextlib import contextmanager,redirect_stderr,redirect_stdout
from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT

@contextmanager
def redirect_out():
    with open("log.txt", 'w') as file:
        with redirect_stderr(file) as err, redirect_stdout(file) as out:
            yield (err, out)

if (len(sys.argv) != 3):
    print(f"{sys.argv[0]}: Improper arguments: {sys.argv}\nUsage: {sys.argv[0]} INPUT OUTPUT", file=sys.stderr)
    exit(1)

load_dotenv()

with redirect_out(): # get_roboflow_model will complain about not having inference[*], so make it shut up
    from inference.models.utils import get_roboflow_model
    model = get_roboflow_model(model_id="frc-scouting-application/2", api_key=os.getenv('ROBOFLOW_API_KEY'))
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Initialise a dictionary to hold previous positions and velocities
object_data = {}

# List to store results for CSV
results_list = []

cap = VideoCapture(sys.argv[1])
totalframes = int(cap.get(CAP_PROP_FRAME_COUNT))

currentframe = 0
starttime = time()
frametime = floor(starttime * 1000)
frametimes = []

def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
    global object_data
    global frametime, currentframe, totalframes
    results = model.infer(frame)[0]
    detections = sv.Detections.from_inference(results)
    detections = tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {class_id}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    # Process each detection
    for class_id, tracker_id, bbox in zip(detections.class_id, detections.tracker_id, detections.xyxy):
        current_position = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])  # Centre of the bounding box
        if tracker_id not in object_data:
            object_data[tracker_id] = {
                'previous_position': current_position,
                'velocity': np.array([0.0, 0.0]),
                'acceleration': np.array([0.0, 0.0]),
                'last_frame': frame_index
            }
        else:
            previous_position = object_data[tracker_id]['previous_position']
            time_elapsed = 1  # Assuming frame rate is constant and equals 1 second for simplicity

            # Calculate distance traveled
            distance = np.linalg.norm(current_position - previous_position)
            velocity = distance / time_elapsed

            # Calculate acceleration
            previous_velocity = object_data[tracker_id]['velocity']
            acceleration = (velocity - np.linalg.norm(previous_velocity)) / time_elapsed

            # Update the object data
            object_data[tracker_id]['previous_position'] = current_position
            object_data[tracker_id]['velocity'] = np.array([
                velocity * np.cos(np.arctan2(current_position[1] - previous_position[1], current_position[0] - previous_position[0])),
                velocity * np.sin(np.arctan2(current_position[1] - previous_position[1], current_position[0] - previous_position[0]))
            ])
            object_data[tracker_id]['acceleration'] = acceleration
            object_data[tracker_id]['last_frame'] = frame_index

            # Store results for CSV
            results_list.append({
                'tracker_id': tracker_id,
                'frame_index': frame_index,
                'position_x': current_position[0],
                'position_y': current_position[1],
                'velocity': np.linalg.norm(object_data[tracker_id]['velocity']),
                'acceleration': object_data[tracker_id]['acceleration'],
            })

    annotated_frame = box_annotator.annotate(frame.copy(), detections = detections)
    mstime = floor(time() * 1000)
    print(f"Frame {currentframe}/{totalframes} time: {mstime - frametime}ms{' ' * 15}", end="\r")
    frametimes.append(mstime - frametime)
    frametime = mstime
    currentframe += 1
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)


sv.process_video(
    source_path=sys.argv[1],
    target_path=sys.argv[2],
    callback=callback
)

# Save results to CSV after processing the video
pd.DataFrame(results_list).to_csv('tracking_results.csv', index=False)

print(f"\nTotal runtime: {floor(time() - starttime)} secs\nAverage time: {floor(np.average(frametimes))}ms")
