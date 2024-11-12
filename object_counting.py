import cv2
from ultralytics import YOLO

paused = False

# Load the YOLO model
model = YOLO('/home/gles/Praktikum/MachineTraining/my_proj/train6/weights/best.pt')

# Open the video file
video_path = "drone_2.MP4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    if not paused:    
        success, frame = cap.read()
        if success:
            
            # Resize image
            scale_percent = 60  # percent of original size
            width = int(frame.shape[1] * scale_percent / 100)
            height = int(frame.shape[0] * scale_percent / 100)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            
            # Run YOLO detection on the frame
            results = model.predict(frame, show=False, show_conf = False, show_labels = False, line_width = 30)  
            
            # Get the annotated frame
            annotated_frame = results[0].plot(conf = False, line_width = 2, labels =False)
            
            # Get total detections
            total_detections = len(results[0].boxes)
            
            # Add text for total detections (using similar format from documentation)
            text = f"Total Detections: {total_detections}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
            gap = 10
            
            # Draw white background rectangle
            cv2.rectangle(
                annotated_frame,
                (20 - gap, 70 - text_size[1] - gap),
                (20 + text_size[0] + gap, 70 + gap),
                (255, 255, 255),
                -1,
            )
            # Add black text
            cv2.putText(annotated_frame, text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # Display the frame
            cv2.imshow("YOLO Detection", annotated_frame)
            
    key = cv2.waitKey(1)
    if key & 0xFF == ord("q"):  # Quit if 'q' is pressed
        break
    if key & 0xFF == ord('p'):  # Toggle pause with 'p' key
        paused = not paused

cap.release()
cv2.destroyAllWindows()
