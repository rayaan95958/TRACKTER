#COMMANDS:
#s-screenshot
#p-pause
#r-rewind 5 seconds
#f-fast forward 5 seconds
#q-quit

import cv2
import os

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def main(video_path, crop_coords, speed_up_factor=1.5, rewind_seconds=5, fast_forward_seconds=5):
    # Get the video file name without extension and create a folder
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    create_folder(video_name)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps / speed_up_factor)  # Adjust delay to speed up video
    rewind_frames = int(fps * rewind_seconds)
    fast_forward_frames = int(fps * fast_forward_seconds)

    screenshot_count = 1
    paused = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Reached end of video or failed to read the video.")
                break

            # Display the video frame
            cv2.imshow('Video', frame)

        # Wait for key press
        key = cv2.waitKey(delay) & 0xFF

        if key == ord('s'):
            # Crop the frame according to the specified coordinates
            x1, y1, x2, y2 = crop_coords
            cropped_frame = frame[y1:y2, x1:x2]

            # Save the cropped frame as a screenshot
            screenshot_name = os.path.join(video_name, f'img{screenshot_count}.png')
            cv2.imwrite(screenshot_name, cropped_frame)
            print(f"Screenshot saved: {screenshot_name}")
            screenshot_count += 1
        elif key == ord('p'):
            # Pause/resume the video
            paused = not paused
            if paused:
                print("Video paused. Press 'p' again to resume.")
            else:
                print("Video resumed.")
        elif key == ord('r'):
            # Rewind the video by 5 seconds
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = max(0, current_frame - rewind_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"Rewound to frame {new_frame}.")
        elif key == ord('f'):
            # Fast forward the video by 5 seconds
            current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            new_frame = min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), current_frame + fast_forward_frames)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame)
            print(f"Fast forwarded to frame {new_frame}.")
        elif key == ord('q'):
            # Quit the video window
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = r'C:\Users\satar\OneDrive\Desktop\TRACKTER\DATA\PRODUCT VIDEOS\73297-218-50.mp4'  # Replace with your video file path
    crop_coords = (65, 0, 500, 450)  # Replace with your crop coordinates (x1, y1, x2, y2)
    main(video_path, crop_coords, speed_up_factor=1.5, rewind_seconds=5, fast_forward_seconds=5)
