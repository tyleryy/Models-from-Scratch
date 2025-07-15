import cv2
import glob
import os

# Set your image directory and output video path
image_dir = "predicted_flows"  # Change to your directory
output_video = "predicted_flow_video.mp4"
fps = 24  # Frames per second

# Get sorted list of image files (assumes files are named in order, e.g., flow_00001.png)
image_files = sorted(glob.glob(os.path.join(image_dir, "*.png")))

# Read the first image to get frame size
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# Define the video writer with mp4v codec
video = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

# Write each image to the video
for filename in image_files:
    img = cv2.imread(filename)
    video.write(img)

video.release()
print(f"Video saved as {output_video}")