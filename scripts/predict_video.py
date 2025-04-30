import cv2
import os
import random
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('C:/Users/MSJAIUser/PycharmProjects/Senior Research Project/cicada_yolo/runs/detect/train11/weights/best.pt')

# Define the folder containing videos
video_folder = 'G:/photos'
video_paths = [os.path.join(video_folder, vid) for vid in os.listdir(video_folder) if vid.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# Select a random video from the folder
if video_paths:
    selected_video = random.choice(video_paths)
    print(f"Selected video: {selected_video}")

    # Perform prediction on the selected video
    results = model.predict(source=selected_video, save=True, conf=0.75)  # Adjust confidence threshold if needed

    print(f"Predictions saved for {selected_video}")
else:
    print("No videos found in the specified folder.")