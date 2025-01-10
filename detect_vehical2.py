from ultralytics import YOLO
import cv2
import os

# File paths
input_video = r'E:\projects\vehicle_dect\2103099-uhd_3840_2160_30fps.mp4'
output_video = r'E:\projects\vehicle_dect\output_video_2.mp4'

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
