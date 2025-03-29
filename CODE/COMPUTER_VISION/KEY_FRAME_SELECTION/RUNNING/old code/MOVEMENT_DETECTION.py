#REDUNDANT IF USING OBJECT TRACKING ALGORITHM FOR KFS

import cv2
import numpy as np
import time

# Hardcoded path to the video file
video_path = '/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/RUNNING/test.mp4'

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Parameters for background subtraction
history = 500  # Number of frames for the history
varThreshold = 16  # Threshold for detecting shadows
detectShadows = True  # Enable shadow detection

# Create background subtractor object (KNN is more robust to lighting changes)
backSub = cv2.createBackgroundSubtractorKNN(history=history, dist2Threshold=400, detectShadows=detectShadows)

# Define crop margins (in pixels)
left_margin = 240
right_margin = 230
top_margin = 100
bottom_margin = 10

# Set target FPS
target_fps = 10
delay = 1.0 / target_fps  # Delay in seconds between frames to achieve 10 FPS

# Status tracking variables
status = "Stopped"
movement_threshold = 66000  # Threshold for movement detection
prev_brightness = None  # Variable to store previous frame brightness
brightness_threshold = 30  # Threshold for detecting global lighting changes

# Kernel for morphological operations
kernel_size = (5, 5)
kernel = np.ones(kernel_size, np.uint8)

while True:
    start_time = time.time()  # Start time for the frame

    ret, frame = cap.read()
    if not ret:
        break

    # Crop the current frame
    frame_cropped = frame[top_margin:-bottom_margin, left_margin:-right_margin]

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to reduce the impact of lighting changes
    equalized_frame = cv2.equalizeHist(gray_frame)

    # Apply Gaussian blur to smooth small lighting fluctuations
    blurred_frame = cv2.GaussianBlur(equalized_frame, (5, 5), 0)

    # Apply background subtraction
    fg_mask = backSub.apply(blurred_frame)

    # Apply morphological operations to clean up the foreground mask
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)

    # Calculate the current frame's brightness
    current_brightness = np.mean(gray_frame)

    # If there's a global lighting change, ignore this frame
    if prev_brightness is not None and abs(current_brightness - prev_brightness) > brightness_threshold:
        print("Global lighting change detected, ignoring this frame")
        prev_brightness = current_brightness
        continue

    prev_brightness = current_brightness

    # Calculate the amount of movement by summing the white pixels in the foreground mask
    movement_amount = np.sum(fg_mask)
    
    # Debugging output
    print(f"Movement Amount: {movement_amount}")

    # Check movement and update status
    if movement_amount > movement_threshold:
        status = "Moving"
    else:
        status = "Stopped"
    
    # Print status
    print(f"Status: {status}")

    # Display the result
    cv2.imshow('Frame', frame_cropped)
    cv2.imshow('Foreground Mask', fg_mask)

    # Calculate time taken for the frame processing
    elapsed_time = time.time() - start_time

    # Sleep to maintain the target FPS
    time.sleep(max(0, delay - elapsed_time))

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
