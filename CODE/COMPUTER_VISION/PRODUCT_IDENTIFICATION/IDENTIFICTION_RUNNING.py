import os
import time
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import datetime

# Configuration
MODEL_PATH = '/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/PRODUCT_IDENTIFICATION/resnet18_model.pth'
DATASET_DIR = '/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/DATA/DATASET_UNANNOTATED/train'
KEYFRAMES_DIR = '/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/RUNNING/output/keyframes'
OUTPUT_FILE = '/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/DATABASE_MODEL/Identified_parts.txt'
PROCESS_INTERVAL = 10  # seconds between checks when folder is empty

# Get class names from dataset directory
class_names = sorted([d for d in os.listdir(DATASET_DIR) 
                    if os.path.isdir(os.path.join(DATASET_DIR, d))])

# Load the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
model.to(device)

# Image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def get_sorted_keyframes():
    """Get keyframe files sorted by creation time (oldest first)"""
    try:
        files = [os.path.join(KEYFRAMES_DIR, f) for f in os.listdir(KEYFRAMES_DIR) 
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        # Sort by creation time
        files.sort(key=lambda x: os.path.getctime(x))
        return files
    except Exception as e:
        print(f"Error reading keyframes directory: {e}")
        return []

def process_image(image_path):
    """Process a single image and return prediction"""
    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(image)
        
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        max_prob, predicted_class = torch.max(probabilities, 0)
        
        return {
            'filename': os.path.basename(image_path),
            'class': class_names[predicted_class.item()],
            'confidence': float(max_prob.item()),
            'timestamp': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")
        return None

def save_result(result):
    """Save result to output file"""
    try:
        # Create output file if it doesn't exist
        if not os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'w') as f:
                f.write("Filename | Class | Confidence | Timestamp\n")
                f.write("----------------------------------------\n")
        
        with open(OUTPUT_FILE, 'a') as f:
            line = f"{result['filename']} | {result['class']} | {result['confidence']:.4f} | {result['timestamp']}\n"
            f.write(line)
    except Exception as e:
        print(f"Error saving results: {e}")

def process_keyframes():
    """Process all keyframes in order and delete after processing"""
    while True:
        keyframes = get_sorted_keyframes()
        
        if not keyframes:
            print(f"No keyframes found. Waiting {PROCESS_INTERVAL} seconds...")
            time.sleep(PROCESS_INTERVAL)
            continue
        
        for image_path in keyframes:
            try:
                print(f"Processing: {os.path.basename(image_path)}")
                result = process_image(image_path)
                
                if result:
                    save_result(result)
                    print(f"Identified as: {result['class']} (confidence: {result['confidence']:.2f})")
                
                # Delete the file after processing
                os.remove(image_path)
                print(f"Deleted: {os.path.basename(image_path)}")
                
            except Exception as e:
                print(f"Error processing/deleting {image_path}: {e}")
                # Skip to next file if error occurs
        
        # Small delay before checking for new files
        time.sleep(1)

class KeyframeHandler(FileSystemEventHandler):
    def on_created(self, event):
        """Trigger immediate processing when new file is detected"""
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"New keyframe detected: {os.path.basename(event.src_path)}")
            # The main processing loop will pick it up automatically

def start_monitoring():
    """Start monitoring the keyframes directory"""
    event_handler = KeyframeHandler()
    observer = Observer()
    observer.schedule(event_handler, KEYFRAMES_DIR, recursive=False)
    observer.start()
    
    try:
        print(f"Starting keyframe processor. Monitoring: {KEYFRAMES_DIR}")
        print(f"Results will be saved to: {OUTPUT_FILE}")
        print("Press Ctrl+C to stop...")
        process_keyframes()
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Create keyframes directory if it doesn't exist
    os.makedirs(KEYFRAMES_DIR, exist_ok=True)
    
    start_monitoring()