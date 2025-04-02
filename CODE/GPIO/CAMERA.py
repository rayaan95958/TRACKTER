from picamera2 import Picamera2
import cv2
import time
import subprocess

# Initialize the camera
picam2 = Picamera2()

# Configure the camera
config = picam2.create_video_configuration(
    main={"size": (1280, 720)},  # Main stream for recording
    lores={"size": (640, 480)},  # Low-res stream for preview
    display="lores"
)
picam2.configure(config)

# Function to convert H264 to MP4
def convert_to_mp4(h264_file, mp4_file):
    cmd = ["ffmpeg", "-y", "-i", h264_file, "-c:v", "copy", mp4_file]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Start camera preview
picam2.start()

# Start recording loop
video_counter = 1
recording_duration = 60  # First video: 1 minute (60 sec), then every 10 minutes (600 sec)
start_time = time.time()

while True:
    output_h264 = f"/home/pi/Videos/video_{video_counter}.h264"
    output_mp4 = f"/home/pi/Videos/video_{video_counter}.mp4"

    # Start recording
    picam2.start_recording(picam2.video_configuration, output_h264)
    print(f"Recording started: {output_h264}")

    time.sleep(recording_duration)  # Wait for the duration

    # Stop recording
    picam2.stop_recording()
    print(f"Recording stopped: {output_h264}")

    # Convert to MP4
    convert_to_mp4(output_h264, output_mp4)
    print(f"Converted to MP4: {output_mp4}")

    # Remove H264 file to save space
    subprocess.run(["rm", output_h264])

    # Update counter and duration
    video_counter += 1
    recording_duration = 600  # Next videos: 10 minutes (600 sec)
