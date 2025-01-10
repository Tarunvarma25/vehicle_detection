
# Vehicle and Object Detection with YOLOv8

This project demonstrates real-time vehicle and object detection using the YOLOv8 model. It processes video input, detects vehicles like cars, trucks, buses and motorcycles and outputs a processed video with bounding boxes and confidence scores.

---

## Features

- Detects vehicles (`car`, `truck`, `bus`, `motorcycle`).
- Processes videos and adds bounding boxes with class labels and confidence scores.
- Adjustable confidence threshold for fine-tuning detection sensitivity.
- Supports customization to include additional object classes.

---

## Requirements

### Install Dependencies
Install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Dependencies
- **ultralytics**: For YOLOv8 model integration.
- **opencv-python**: For video processing.

---

## How to Run the Project

### Step 1: Download the YOLOv8 Model
The script uses the YOLOv8n model by default. You can download the pretrained YOLOv8 weights (e.g., `yolov8n.pt`) from [Ultralytics](https://github.com/ultralytics/ultralytics).

### Step 2: Place Your Video File
Save your video file in the desired directory. Update the `input_video` and `output_video` paths in the script.

For example:
```python
input_video = r'E:\projects\vehicle_dect\2099536-hd_1920_1080_30fps.mp4'
output_video = r'E:\projects\vehicle_dect\output_video.mp4'
```

### Step 3: Run the Script
Run the Python script:
```bash
python detect_vehicles.py
```

### Step 4: Check Output
The output video (with bounding boxes and labels) will be saved in the specified output path:
```
E:\projects\vehicle_dect\output_video.mp4
```

---

## File Structure

```plaintext
vehicle_detection_project/
│
├── detect_vehicles.py    # Main Python script for detection
├── requirements.txt      # List of dependencies
├── README.md             # Project documentation
├── input_video.mp4       # Input video file (example path)
└── output_video.mp4      # Processed output video (example path)
```

---

## Key Parameters

- **Confidence Threshold**: Adjust `conf=0.3` in the script to control detection sensitivity.
- **Classes**: The script currently detects:
  - `car`
  - `truck`
  - `bus`
  - `motorcycle`

You can add or modify the classes to detect more objects by updating this section:
```python
if model.names[class_id] in ["car", "truck", "bus", "motorcycle"]:
```

---

## Sample Output

For an input video containing traffic and pedestrians:
- Bounding boxes will appear around detected vehicles.
- Each box will display the object type (e.g., `car`, `bus`) and confidence score.

---

## Troubleshooting

### Video Not Opening
- Ensure the input video path is correct and accessible.
- Use raw string format (`r"path_to_video"`) for file paths on Windows.
- Convert the video to a standard format (e.g., `.mp4`) using FFmpeg:
  ```bash
  ffmpeg -i input_video.avi -vcodec libx264 -crf 22 output_video.mp4
  ```

### Objects Not Detected
- Lower the confidence threshold in the script:
  ```python
  results = model.predict(frame, conf=0.3, show=False)
  ```
- Ensure the YOLOv8 model (`yolov8n.pt`) is correctly downloaded.

---

## Customization

### Add More Classes
To include additional objects, update the class list:
```python
if model.names[class_id] in ["car", "truck", "bus", "motorcycle"]:
```

## License

This project is licensed under the MIT License. Feel free to use and modify it as needed.

---

## Acknowledgments

- **[Ultralytics](https://github.com/ultralytics/ultralytics)** for YOLOv8.
- **OpenCV** for video processing.

---

