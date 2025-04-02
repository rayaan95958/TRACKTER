import subprocess

# Define paths to scripts
buzzer_lights_script = "/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/GPIO/BUZZER_LIGHTS.py"
camera_script = "/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/GPIO/CAMERA.py"

# Run BUZZER_LIGHTS.py and wait for it to finish
subprocess.run(["python3", buzzer_lights_script])

# Run CAMERA.py in the background (keeps running)
subprocess.Popen(["python3", camera_script])

# Master script exits, but CAMERA.py keeps running
