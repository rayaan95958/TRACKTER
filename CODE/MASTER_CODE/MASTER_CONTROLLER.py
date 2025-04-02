import subprocess
import threading
import time
import signal
import sys
from pathlib import Path

# Configuration
SCRIPTS = {
    "detection_tracking": Path("/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/RUNNING/DETECTION_TRACKING_KFS_VIDEO.py"),
    "identification": Path("/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/PRODUCT_IDENTIFICATION/IDENTIFICTION_RUNNING.py")
}

class ScriptRunner(threading.Thread):
    def __init__(self, script_path):
        threading.Thread.__init__(self)
        self.script_path = script_path
        self.process = None
        self.stop_event = threading.Event()
        self.daemon = True  # Terminates when main thread exits

    def run(self):
        while not self.stop_event.is_set():
            try:
                self.process = subprocess.Popen(
                    ["python3", str(self.script_path)],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True
                )
                print(f"Started {self.script_path.name} (PID: {self.process.pid})")
                
                # Monitor process output
                for line in self.process.stdout:
                    print(f"[{self.script_path.stem}] {line.strip()}")
                
                # Wait for process completion unless stopped
                while not self.stop_event.is_set():
                    if self.process.poll() is not None:
                        break
                    time.sleep(0.5)
                
                if not self.stop_event.is_set():
                    print(f"Process {self.script_path.name} terminated, restarting...")
                    time.sleep(2)  # Prevent rapid restart loops
                    
            except Exception as e:
                print(f"Error running {self.script_path.name}: {str(e)}")
                time.sleep(5)

    def stop(self):
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
        print(f"Stopped {self.script_path.name}")

def signal_handler(sig, frame):
    print("\nShutting down gracefully...")
    for runner in runners.values():
        runner.stop()
    sys.exit(0)

if __name__ == "__main__":
    # Verify scripts exist
    for name, path in SCRIPTS.items():
        if not path.exists():
            print(f"Error: Script not found at {path}")
            sys.exit(1)

    # Initialize runners
    runners = {
        name: ScriptRunner(path)
        for name, path in SCRIPTS.items()
    }

    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Start all processes
    print("Starting all subsystems...")
    for runner in runners.values():
        runner.start()

    # Main monitoring loop
    while True:
        time.sleep(1)
        for name, runner in runners.items():
            if not runner.is_alive() and not runner.stop_event.is_set():
                print(f"Warning: {name} thread died, attempting to restart...")
                runners[name] = ScriptRunner(SCRIPTS[name])
                runners[name].start()