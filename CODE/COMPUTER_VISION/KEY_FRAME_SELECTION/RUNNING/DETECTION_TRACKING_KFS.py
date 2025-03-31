#USES STATIC CONVEYER BELT SPEED AND HAS KFS LOGIC
#INCREASE FPS IF TRACKING BEING FAULTY?

import subprocess
import cv2
import os
import numpy as np
import re
from scipy.optimize import linear_sum_assignment
cv2.setUseOptimized(True)

# ===== Configuration =====
darknet_path = "/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/RUNNING/darknet/darknet/darknet"
cfg_path = "/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/RUNNING/darknet/darknet/cfg/yolov4-tiny-custom.cfg"
weights_path = "/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/TRAINING/yolov4-tiny/training/yolov4-tiny-custom_best.weights"
data_path = "/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/RUNNING/darknet/darknet/data/obj.data"
video_path = "/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/RUNNING/test.mp4"
output_dir = "/mnt/c/Users/satar/OneDrive/Desktop/TRACKTER/CODE/COMPUTER_VISION/KEY_FRAME_SELECTION/RUNNING/output"
keyframes_dir = os.path.join(output_dir, "keyframes")

# Tracking parameters
min_detection_confidence = 0.4
max_age_since_last_detection = 5
min_track_points = 8
max_distance = 50
reid_threshold = 0.7

# Key frame selection parameters
MIDDLE_X_RANGE = (0.4, 0.6)  # 40-60% of frame width considered "middle"
MIN_KEYFRAME_CONFIDENCE = 0.5  # Minimum confidence for keyframe selection

# Conveyor-specific parameters
CONVEYOR_SPEED = -10  # Negative for right-to-left movement (pixels/frame)
STALL_THRESHOLD = 3  # Pixels/frame below which we consider a stall
STALL_MAX_AGE = 15  # Frames to keep tracks during stalls

lk_params = dict(winSize=(25, 25),  # Larger window for linear motion
                maxLevel=1,         # Fewer levels for flat conveyor
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# ===== Initialization =====
def verify_paths():
    if not os.path.exists(darknet_path):
        raise FileNotFoundError(f"Darknet binary not found at {darknet_path}")
    required_files = {
        "Config": cfg_path,
        "Weights": weights_path,
        "Data": data_path,
        "Video": video_path
    }
    for name, path in required_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} file not found at {path}")
    if not os.access(darknet_path, os.X_OK):
        os.chmod(darknet_path, 0o755)

verify_paths()
os.makedirs(output_dir, exist_ok=True)
os.makedirs(keyframes_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file {video_path}")

fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_interval = 3*fps  # Processing rate 
print(f"Video: {width}x{height} @ {fps}fps")
print(f"Processing at 1 frame every 3s (processing every {frame_interval} frames)")

tracked_objects = {}
next_id = 1
prev_frame = None
frame_count = 0
conveyor_moving = True

# Set to keep track of objects that already have keyframes
objects_with_keyframes = set()

# ===== Core Functions =====
def parse_detection_output(output):
    detections = []
    pattern = r'(\w+):\s*(\d+)%\s*\(left_x:\s*(\d+)\s*top_y:\s*(\d+)\s*width:\s*(\d+)\s*height:\s*(\d+)\)'
    matches = re.findall(pattern, output)
    for match in matches:
        try:
            confidence = float(match[1]) / 100
            if confidence >= min_detection_confidence:
                detections.append({
                    'class': match[0],
                    'confidence': confidence,
                    'bbox': [int(match[2]), int(match[3]), int(match[4]), int(match[5])]
                })
        except Exception as e:
            print(f"Error parsing detection: {e}")
    print(f"Found {len(detections)} valid detections")
    return detections

def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1+w1, x2+w2)
    yi2 = min(y1+h1, y2+h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    return inter_area / float(box1_area + box2_area - inter_area)

def detect_conveyor_movement(prev_frame, current_frame):
    """Detect if conveyor is moving using optical flow"""
    if prev_frame is None:
        return True
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate average flow in horizontal direction
    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    avg_flow_x = np.mean(flow[...,0])
    
    # Conveyor is moving if average flow exceeds threshold
    return abs(avg_flow_x) > STALL_THRESHOLD

def predict_conveyor_motion(tracks):
    """Apply conveyor motion prior to predictions"""
    for track_id in tracks:
        x, y, w, h = tracks[track_id]['bbox']
        if conveyor_moving:
            # Apply constant speed prediction (right-to-left)
            tracks[track_id]['predicted_bbox'] = (
                x + CONVEYOR_SPEED,  # Moves left (negative direction)
                y, w, h
            )
        else:
            # During stall, keep same position
            tracks[track_id]['predicted_bbox'] = (x, y, w, h)

def match_detections_to_tracks(detections, tracked_objects):
    matches = {}
    unmatched_detections = list(range(len(detections))) if detections else []
    unmatched_tracks = list(tracked_objects.keys()) if tracked_objects else []
    
    if not detections or not tracked_objects:
        return matches, unmatched_detections, unmatched_tracks
    
    cost_matrix = np.zeros((len(tracked_objects), len(detections)))
    for i, (track_id, obj) in enumerate(tracked_objects.items()):
        for j, det in enumerate(detections):
            # Use predicted position if available
            track_pos = obj.get('predicted_bbox', obj['bbox'])
            track_center = np.array([track_pos[0] + track_pos[2]/2, 
                                    track_pos[1] + track_pos[3]/2])
            det_center = np.array([det['bbox'][0] + det['bbox'][2]/2,
                                  det['bbox'][1] + det['bbox'][3]/2])
            cost_matrix[i,j] = np.linalg.norm(track_center - det_center)
    
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    
    for i, j in zip(row_ind, col_ind):
        track_id = list(tracked_objects.keys())[i]
        det = detections[j]
        iou = calculate_iou(tracked_objects[track_id]['bbox'], det['bbox'])
        
        if iou > reid_threshold or cost_matrix[i,j] < max_distance:
            matches[track_id] = j
    
    unmatched_detections = [j for j in range(len(detections)) if j not in matches.values()]
    unmatched_tracks = [track_id for track_id in tracked_objects if track_id not in matches]
    
    return matches, unmatched_detections, unmatched_tracks

def is_in_middle_screen(bbox, frame_width):
    """Check if object is roughly in the middle of the screen"""
    x, y, w, h = bbox
    center_x = x + w/2
    middle_start = frame_width * MIDDLE_X_RANGE[0]
    middle_end = frame_width * MIDDLE_X_RANGE[1]
    return middle_start <= center_x <= middle_end

def should_capture_keyframe(track_id, bbox, confidence):
    """Determine if we should capture a keyframe for this object"""
    # Skip if we already have a keyframe for this object
    if track_id in objects_with_keyframes:
        return False
    
    # Check confidence threshold
    if confidence < MIN_KEYFRAME_CONFIDENCE:
        return False
    
    # Check if object is in middle of screen
    return is_in_middle_screen(bbox, width)

def save_keyframe(frame, track_id, bbox, class_name, frame_count):
    """Save keyframe with object highlighted"""
    x, y, w, h = bbox
    
    # Create a copy of the frame with the object highlighted
    keyframe = frame.copy()
    cv2.rectangle(keyframe, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.putText(keyframe, f"ID: {track_id} {class_name}", (x, y-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    # Save the keyframe
    filename = os.path.join(keyframes_dir, f"keyframe_{track_id}_{frame_count:06d}.jpg")
    cv2.imwrite(filename, keyframe)
    
    # Mark this object as having a keyframe
    objects_with_keyframes.add(track_id)
    
    print(f"Saved keyframe for ID {track_id} at frame {frame_count}")

# ===== Main Processing Loop =====
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        print(f"\nProcessing frame {frame_count}")
        
        # Detect conveyor movement state
        conveyor_moving = detect_conveyor_movement(prev_frame, frame)
        current_max_age = STALL_MAX_AGE if not conveyor_moving else max_age_since_last_detection
        
        # Save frame for YOLO
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        
        # Run YOLO detection (MODIFIED COMMAND)
        cmd = [
            darknet_path,
            'detector', 'test', 
            data_path, cfg_path, weights_path,
            frame_filename,
            '-thresh', str(min_detection_confidence),
            '-ext_output', '-dont_show',
            '-dont_show_av_time',
            '-time_limit_sec', '2.5'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        detections = parse_detection_output(result.stdout)
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is None:
            prev_frame = frame.copy()
            prev_gray = current_gray.copy()
            
            for det in detections:
                x, y, w, h = det['bbox']
                mask = np.zeros_like(prev_gray)
                mask[y:y+h, x:x+w] = 255
                points = cv2.goodFeaturesToTrack(prev_gray, mask=mask,
                                               maxCorners=100,
                                               qualityLevel=0.3,
                                               minDistance=7)
                if points is not None and len(points) >= min_track_points:
                    tracked_objects[next_id] = {
                        'bbox': det['bbox'],
                        'points': points,
                        'class': det['class'],
                        'age': 1,
                        'detected': True,
                        'history': [det['bbox']],
                        'confidence': det['confidence']
                    }
                    
                    # Check if we should capture a keyframe for this new object
                    if should_capture_keyframe(next_id, det['bbox'], det['confidence']):
                        save_keyframe(frame, next_id, det['bbox'], det['class'], frame_count)
                    
                    next_id += 1
        else:
            # Predict movement based on conveyor state
            predict_conveyor_motion(tracked_objects)
            
            # Update existing tracks
            for obj_id in list(tracked_objects.keys()):
                if conveyor_moving:
                    # Use optical flow when conveyor is moving
                    old_points = tracked_objects[obj_id]['points']
                    new_points, status, _ = cv2.calcOpticalFlowPyrLK(
                        prev_gray, current_gray, old_points, None, **lk_params)
                    
                    good_new = new_points[status.flatten() == 1]
                    
                    if len(good_new) >= min_track_points:
                        # Blend optical flow with conveyor prediction
                        flow_x = np.median(good_new[:,0,0] - old_points[status.flatten()==1][:,0,0])
                        pred_x = tracked_objects[obj_id].get('predicted_bbox', tracked_objects[obj_id]['bbox'])[0]
                        blended_x = int(0.7*pred_x + 0.3*(tracked_objects[obj_id]['bbox'][0] + flow_x))
                        
                        tracked_objects[obj_id].update({
                            'points': good_new.reshape(-1, 1, 2),
                            'bbox': (blended_x,
                                    tracked_objects[obj_id]['bbox'][1],
                                    tracked_objects[obj_id]['bbox'][2],
                                    tracked_objects[obj_id]['bbox'][3]),
                            'age': tracked_objects[obj_id]['age'] + 1,
                            'detected': False
                        })
                    else:
                        del tracked_objects[obj_id]
                else:
                    # During stall, just update age
                    tracked_objects[obj_id]['age'] += 1
            
            # Match detections to tracks
            matches, unmatched_dets, unmatched_tracks = match_detections_to_tracks(detections, tracked_objects)
            
            # Update matched tracks
            for track_id, det_idx in matches.items():
                det = detections[det_idx]
                tracked_objects[track_id].update({
                    'bbox': det['bbox'],
                    'class': det['class'],
                    'detected': True,
                    'age': 1,  # Reset age counter
                    'confidence': det['confidence']
                })
                
                # Refresh tracking points
                x, y, w, h = det['bbox']
                mask = np.zeros_like(current_gray)
                mask[y:y+h, x:x+w] = 255
                points = cv2.goodFeaturesToTrack(current_gray, mask=mask,
                                               maxCorners=100,
                                               qualityLevel=0.3,
                                               minDistance=7)
                if points is not None:
                    tracked_objects[track_id]['points'] = points
                
                # Check if we should capture a keyframe for this object
                if should_capture_keyframe(track_id, det['bbox'], det['confidence']):
                    save_keyframe(frame, track_id, det['bbox'], det['class'], frame_count)
            
            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                det = detections[det_idx]
                x, y, w, h = det['bbox']
                mask = np.zeros_like(current_gray)
                mask[y:y+h, x:x+w] = 255
                points = cv2.goodFeaturesToTrack(current_gray, mask=mask,
                                               maxCorners=100,
                                               qualityLevel=0.3,
                                               minDistance=7)
                if points is not None and len(points) >= min_track_points:
                    tracked_objects[next_id] = {
                        'bbox': det['bbox'],
                        'points': points,
                        'class': det['class'],
                        'age': 1,
                        'detected': True,
                        'history': [det['bbox']],
                        'confidence': det['confidence']
                    }
                    
                    # Check if we should capture a keyframe for this new object
                    if should_capture_keyframe(next_id, det['bbox'], det['confidence']):
                        save_keyframe(frame, next_id, det['bbox'], det['class'], frame_count)
                    
                    next_id += 1
            
            # Remove old undetected tracks
            for track_id in unmatched_tracks:
                if tracked_objects[track_id]['age'] > current_max_age:
                    del tracked_objects[track_id]
                    # Also remove from keyframe tracking if it was there
                    if track_id in objects_with_keyframes:
                        objects_with_keyframes.remove(track_id)
        
        # Visualization - ensure bounding boxes with IDs are drawn
        output_frame = frame.copy()
        for obj_id, obj in tracked_objects.items():
            if not obj['bbox']:
                continue
                
            x, y, w, h = obj['bbox']
            color = (0, 255, 0) if obj['detected'] else (0, 0, 255)
            thickness = 2
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), color, thickness)
            
            # Draw tracking ID and status
            status = "D" if obj['detected'] else "P"
            has_keyframe = "K" if obj_id in objects_with_keyframes else ""
            label = f"ID:{obj_id} {obj['class']} {status}{has_keyframe}"
            cv2.putText(output_frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save output with tracking information
        tracked_filename = os.path.join(output_dir, f"frame_{frame_count:04d}_tracked.jpg")
        cv2.imwrite(tracked_filename, output_frame)
        
        prev_frame = frame.copy()
        prev_gray = current_gray.copy()

    # Garbage collection
    if frame_count % 100 == 0:
        import gc
        gc.collect()

    frame_count += 1

cap.release()
print(f"\nProcessing complete. Final track count: {next_id-1}")
print(f"Keyframes saved in: {keyframes_dir}")
print(f"Total keyframes captured: {len(objects_with_keyframes)}")