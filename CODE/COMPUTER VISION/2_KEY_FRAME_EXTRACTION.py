import cv2
import numpy as np
import time  

def select_key_frames(video_path, intensity_threshold=150, center_x_threshold=50, center_y_threshold=50, delay_sec=0.5):
    try:
        cap = cv2.VideoCapture(video_path)

        # Check if video capture is open
        if not cap.isOpened():
            print("Error: Unable to open video.")
            return None

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        key_frame = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to grayscale for intensity-based processing
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply intensity thresholding
            _, binary = cv2.threshold(gray, intensity_threshold, 255, cv2.THRESH_BINARY)

            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Process contours and select key frame
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

                # Calculate bounding box center
                center_x = x + w // 2
                center_y = y + h // 2

                # Check if bounding box is centered
                if (abs(center_x - frame_width // 2) < center_x_threshold and
                        abs(center_y - frame_height // 2) < center_y_threshold):
                    # Draw bounding box on frame
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Set key frame
                    key_frame = frame.copy()

                    # Display the frame with bounding box (optional)
                    cv2.imshow('Key Frame Selection', frame)
                    cv2.waitKey(1)  # Display frame for a short period

                    # Delay for specified seconds
                    time.sleep(delay_sec)
            
            # Break out of loop if key frame is found
            if key_frame is not None:
                break

        cap.release()
        cv2.destroyAllWindows()

        return key_frame

    except Exception as e:
        print(f"Error: {e}")
        return None

# Example usage
video_path = r'C:\Users\satar\OneDrive\Desktop\TRACKTER\CODE\sample_video.mp4'
key_frame = select_key_frames(video_path)

# Optionally, save key frame
if key_frame is not None:
    cv2.imwrite("key_frame.jpg", key_frame)
