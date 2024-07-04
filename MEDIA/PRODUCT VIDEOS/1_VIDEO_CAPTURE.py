#75 inches away from products 

import cv2
import time

# Define the duration in seconds
duration = 3600  # 1 hour

# Define the filename and codec
filename = '73296-023-51.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4

# Open the default camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device.")
    exit()

# Get the width and height of the frame
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Define the VideoWriter object
out = cv2.VideoWriter(filename, fourcc, 20.0, (frame_width, frame_height))

# Record video for the specified duration
start_time = time.time()
while int(time.time() - start_time) < duration:
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        # Optionally, display the frame
        cv2.imshow('Recording...', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when the recording is done
cap.release()
out.release()
cv2.destroyAllWindows()
