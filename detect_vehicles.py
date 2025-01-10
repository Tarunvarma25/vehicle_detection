# from ultralytics import YOLO
# import cv2

# # Load the YOLOv8 model
# model = YOLO('yolov8n.pt')  # Use YOLOv8 pretrained weights (e.g., yolov8n.pt, yolov8s.pt)

# # Function to process video and detect different vehicles
# def detect_vehicles(video_path, output_path):
#     # Open the input video file
#     cap = cv2.VideoCapture(video_path)
    
#     # Get video properties
#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fps = int(cap.get(cv2.CAP_PROP_FPS))
#     frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#     # Define codec and create a VideoWriter object for output video
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

#     # Process the video frame by frame
#     print(f"Processing video: {video_path} ({frame_count} frames)")
#     frame_idx = 0
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_idx += 1

#         # Run YOLOv8 model on the current frame
#         results = model.predict(frame, conf=0.5, show=False)  # Adjust confidence threshold if needed

#         # Draw bounding boxes for vehicle classes
#         for result in results:
#             boxes = result.boxes  # Bounding boxes
#             for box in boxes:
#                 # Extract bounding box coordinates, confidence score, and class ID
#                 x1, y1, x2, y2 = map(int, box.xyxy[0])
#                 conf = box.conf[0]  # Confidence score
#                 class_id = int(box.cls[0])  # Class ID
#                 label = f"{model.names[class_id]} {conf:.2f}"

#                 # Filter for specific vehicle classes
#                 if model.names[class_id] in ["car", "truck", "bus", "motorcycle","train"]:
#                     # Draw bounding box and label
#                     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box
#                     cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Write the processed frame to the output video
#         out.write(frame)

#         # Optional: Print progress
#         if frame_idx % 50 == 0:
#             print(f"Processed frame {frame_idx}/{frame_count}")

#     # Release resources
#     cap.release()
#     out.release()
#     cv2.destroyAllWindows()
#     print(f"Vehicle detection completed. Output saved to {output_path}")

# # Specify paths for the input video and output video
# input_video = 'E:/projects/vehicle_dect/2099536-hd_1920_1080_30fps.mp4'  # Replace with your video file path
# output_video = 'E:/projects/vehicle_dect/output_video.mp4'  # Replace with your desired output path

# # Run vehicle detection
# detect_vehicles(input_video, output_video)

# print("Processing complete. Check the output video.")
from ultralytics import YOLO
import cv2
import os

# File paths
input_video = r'E:\projects\vehicle_dect\2099536-hd_1920_1080_30fps.mp4'
output_video = r'E:\projects\vehicle_dect\output_video.mp4'

# Check if the input video exists
if not os.path.exists(input_video):
    print(f"Error: Input file '{input_video}' does not exist.")
    exit()

# Open input video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Unable to open input video file.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Input video properties: {width}x{height}, {fps} FPS, {frame_count} frames")

# Create VideoWriter for output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

if not out.isOpened():
    print("Error: Unable to create output video file.")
    cap.release()
    exit()

# Load YOLOv8 model
model = YOLO('yolov8n.pt')

# Process video frame by frame
frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Finished processing all frames.")
        break

    frame_idx += 1
    print(f"Processing frame {frame_idx}/{frame_count}")

    # Run YOLO model
    results = model.predict(frame, conf=0.5, show=False)

    # Draw bounding boxes for vehicles
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            class_id = int(box.cls[0])
            label = f"{model.names[class_id]} {conf:.2f}"

            if model.names[class_id] in ["car", "truck", "bus", "motorcycle","mini van","ambulancs","polic car","people"]:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write frame to output
    out.write(frame)

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Output saved to '{output_video}'")
