import threading
import time
import logging
import cv2
import numpy as np
import torch
import yaml
import os
from queue import Queue, Empty
from torchvision import transforms
from datetime import datetime
import pytz
from collections import defaultdict, deque
from face_alignment.alignment import norm_crop
from face_detection.scrfd.detector import SCRFD
from face_recognition.arcface.model import iresnet_inference
from face_recognition.arcface.utils import compare_encodings, read_features
from face_tracking.byte_tracker.byte_tracker import BYTETracker
from face_tracking.byte_tracker.visualize import plot_tracking
from app.db.database import engine
from app.db.crud import insert_log, init_db, update_daily_summary
from app.db.database import get_db

# Import shared state for GUI communication
try:
    from gui.shared_state import latest_frames, frame_lock, set_latest_frame
    GUI_AVAILABLE = True
except ImportError:
    # Fallback if GUI is not available
    latest_frames = {}
    frame_lock = threading.Lock()
    GUI_AVAILABLE = False
    
    def set_latest_frame(camera_id, frame):
        with frame_lock:
            latest_frames[camera_id] = frame.copy() if frame is not None else None

# Logging setup
logging.basicConfig(
    filename='attendance.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MultiCameraFaceRecognition:
    def __init__(self, config_path="config/camera_config.yaml"):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        # Initialize models
        self._init_models()
        
        # Load face database
        self.images_names, self.images_embs = read_features("./datasets/face_features/feature")
        print(f"[INFO] Loaded {len(self.images_names)} known faces")
        
        # Threading and data structures
        self.camera_data = {}
        self.data_lock = threading.Lock()
        self.log_queue = Queue()
        self.id_face_mapping = defaultdict(dict)
        self.last_logged_time = {}
        self.recognition_history = defaultdict(lambda: deque(maxlen=3))
        self.running = False
        self.threads = []
        
        # Create snapshots directory
        os.makedirs("snapshots", exist_ok=True)
        
        # Get settings from config
        self.confidence_threshold = self.config['recognition']['confidence_threshold']
        self.logging_interval = self.config['recognition']['logging_interval']
        self.frame_skip = self.config['recognition']['frame_skip']
        self.max_width = self.config['performance']['max_resolution_width']
        
        # Initialize global frames dict
        for cam in self.config['cameras']:
            if cam.get('enabled', True):
                set_latest_frame(cam['camera_id'], np.zeros((480, 640, 3), dtype=np.uint8))
    
    def load_config(self, config_path):
        """Load camera configuration from YAML file"""
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                print(f"[INFO] Loaded configuration from {config_path}")
                return config
        except FileNotFoundError:
            print(f"[ERROR] Configuration file not found: {config_path}")
            return self._default_config()
        except yaml.YAMLError as e:
            print(f"[ERROR] Error parsing YAML file: {e}")
            return self._default_config()
    
    def _default_config(self):
        """Return default configuration if config file is not found"""
        return {
            'cameras': [
                {
                    'camera_id': 'entry_01',
                    'name': 'Entry Camera',
                    'url': 0,
                    'enabled': True,
                    'location': 'Main Entrance',
                    'event_type': 'entry'
                }
            ],
            'recognition': {
                'confidence_threshold': 0.6,
                'logging_interval': 30,
                'frame_skip': 3
            },
            'performance': {
                'max_resolution_width': 640,
                'recognition_interval': 0.5
            },
            'tracking': {
                'track_thresh': 0.6,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'frame_rate': 30,
                'min_box_area': 1000,
                'aspect_ratio_thresh': 1.8
            }
        }
    
    def _init_models(self):
        """Initialize face detection, recognition, and tracking models"""
        try:
            # Initialize face detection model
            self.face_detector = SCRFD(model_file="./face_detection/scrfd/weights/scrfd_10g_bnkps.onnx")
            print("[INFO] Face detector initialized")
            
            # Initialize face recognition model
            self.face_recognizer = iresnet_inference(
                model_name="r100", 
                path="./face_recognition/arcface/weights/arcface_r100.pth", 
                device=self.device
            )
            print("[INFO] Face recognizer initialized")
            
            # Initialize tracking for each camera
            self.trackers = {}
            for cam in self.config['cameras']:
                if cam.get('enabled', True):
                    tracker_config = self.config.get('tracking', {})
                    self.trackers[cam['camera_id']] = BYTETracker(
                        track_thresh=tracker_config.get('track_thresh', 0.6),
                        track_buffer=tracker_config.get('track_buffer', 30),
                        match_thresh=tracker_config.get('match_thresh', 0.8),
                        frame_rate=tracker_config.get('frame_rate', 30)
                    )
            
            print("[INFO] All models initialized successfully")
            
        except Exception as e:
            print(f"[ERROR] Failed to initialize models: {e}")
            raise
    
    def preprocess_frame(self, frame):
        """Preprocess frame for better performance"""
        if frame is None:
            return None
        
        # Resize frame if too large
        height, width = frame.shape[:2]
        if width > self.max_width:
            scale = self.max_width / width
            new_width = int(width * scale)
            new_height = int(height * scale)
            frame = cv2.resize(frame, (new_width, new_height))
        
        return frame
    
    def detect_and_recognize_faces(self, frame, camera_id):
        """Detect and recognize faces in a frame"""
        if frame is None:
            return [], []
        
        try:
            # Detect faces
            bboxes, kpss = self.face_detector.detect(frame, input_size=(640, 640))
            
            if len(bboxes) == 0:
                return [], []
            
            # Prepare data for tracking
            detections = []
            embeddings = []
            
            for i, (bbox, kps) in enumerate(zip(bboxes, kpss)):
                # Extract face region
                face_img = norm_crop(frame, kps, image_size=112)
                
                # Get face embedding
                embedding = self.face_recognizer(face_img)
                embeddings.append(embedding)
                
                # Prepare detection for tracking (x1, y1, x2, y2, confidence)
                detection = [bbox[0], bbox[1], bbox[2], bbox[3], bbox[4]]
                detections.append(detection)
            
            return detections, embeddings
            
        except Exception as e:
            print(f"[ERROR] Face detection/recognition failed for camera {camera_id}: {e}")
            return [], []
    
    def recognize_face(self, embedding):
        """Match face embedding against known faces"""
        if len(self.images_embs) == 0:
            return "Unknown", 0.0
        
        # Compare with known faces
        similarities = compare_encodings(embedding, self.images_embs)
        
        if len(similarities) > 0:
            max_sim = np.max(similarities)
            if max_sim >= self.confidence_threshold:
                max_idx = np.argmax(similarities)
                return self.images_names[max_idx], max_sim
        
        return "Unknown", 0.0
    
    def update_tracking(self, camera_id, detections):
        """Update tracking for a camera"""
        if camera_id not in self.trackers:
            return []
        
        if len(detections) == 0:
            return []
        
        # Convert detections to numpy array
        det_array = np.array(detections)
        
        # Update tracker
        online_targets = self.trackers[camera_id].update(det_array)
        
        return online_targets
    
    def should_log_recognition(self, person_name, camera_id):
        """Check if recognition should be logged based on time interval"""
        current_time = time.time()
        key = f"{person_name}_{camera_id}"
        
        if key not in self.last_logged_time:
            self.last_logged_time[key] = current_time
            return True
        
        if current_time - self.last_logged_time[key] >= self.logging_interval:
            self.last_logged_time[key] = current_time
            return True
        
        return False
    
    def save_snapshot(self, frame, person_name, camera_id):
        """Save snapshot of recognized person"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"snapshots/{person_name}_{camera_id}_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            return filename
        except Exception as e:
            print(f"[ERROR] Failed to save snapshot: {e}")
            return None
    
    def log_recognition(self, person_name, camera_info, confidence, snapshot_path=None):
        """Add recognition to log queue"""
        ist = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(ist)
        
        log_entry = {
            'person_name': person_name,
            'camera_id': camera_info['camera_id'],
            'camera_name': camera_info['name'],
            'location': camera_info.get('location', 'Unknown'),
            'timestamp': timestamp,
            'confidence': confidence,
            'snapshot_path': snapshot_path,
            'event_type': camera_info.get('event_type', 'entry'),
            'tracking_id': None  # Add tracking_id for consistency
        }
        
        self.log_queue.put(log_entry)
        
        # Also log to file
        logging.info(f"Recognized: {person_name} at {camera_info['name']} (confidence: {confidence:.2f})")
    
    def process_camera(self, camera_info):
        """Process frames from a single camera"""
        camera_id = camera_info['camera_id']
        source = camera_info['source']
        
        print(f"[INFO] Starting camera {camera_id} ({camera_info['name']})")
        
        # Initialize camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"[ERROR] Failed to open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    print(f"[WARNING] Failed to read frame from camera {camera_id}")
                    time.sleep(0.1)
                    continue
                
                # Preprocess frame
                frame = self.preprocess_frame(frame)
                if frame is None:
                    continue
                
                # Update shared frame for GUI
                set_latest_frame(camera_id, frame)
                
                # Skip frames for performance
                if frame_count % self.frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Detect and recognize faces
                detections, embeddings = self.detect_and_recognize_faces(frame, camera_id)
                
                if len(detections) > 0:
                    # Update tracking
                    online_targets = self.update_tracking(camera_id, detections)
                    
                    # Process each tracked target
                    for i, (target, embedding) in enumerate(zip(online_targets, embeddings)):
                        if i >= len(embeddings):
                            break
                        
                        # Recognize face
                        person_name, confidence = self.recognize_face(embedding)
                        
                        # Update recognition history for stability
                        track_id = target.track_id
                        self.recognition_history[f"{camera_id}_{track_id}"].append((person_name, confidence))
                        
                        # Get most common recognition
                        history = self.recognition_history[f"{camera_id}_{track_id}"]
                        if len(history) >= 2:  # Wait for at least 2 recognitions
                            names = [h[0] for h in history if h[1] >= self.confidence_threshold]
                            if names:
                                # Get most common name
                                most_common = max(set(names), key=names.count)
                                avg_confidence = np.mean([h[1] for h in history if h[0] == most_common])
                                
                                # Log if needed
                                if (most_common != "Unknown" and 
                                    self.should_log_recognition(most_common, camera_id)):
                                    
                                    # Save snapshot
                                    snapshot_path = self.save_snapshot(frame, most_common, camera_id)
                                    
                                    # Log recognition
                                    self.log_recognition(most_common, camera_info, avg_confidence, snapshot_path)
                    
                    # Draw tracking results
                    frame = plot_tracking(frame, online_targets, frame_id=frame_count)
                
                # Store processed frame
                with self.data_lock:
                    self.camera_data[camera_id] = {
                        'frame': frame.copy(),
                        'detections': len(detections),
                        'timestamp': time.time()
                    }
                
                frame_count += 1
                
        except Exception as e:
            print(f"[ERROR] Error in camera {camera_id} processing: {e}")
            logging.error(f"Camera {camera_id} error: {e}")
        
        finally:
            cap.release()
            print(f"[INFO] Camera {camera_id} stopped")
    
    def database_worker(self):
        """Worker thread for database operations"""
        print("[INFO] Database worker started")
        
        while self.running or not self.log_queue.empty():
            try:
                # Get log entry with timeout
                log_entry = self.log_queue.get(timeout=1)
                
                # Insert into database with proper error handling
                try:
                    with next(get_db()) as db:
                        # Ensure log_entry has all required fields
                        if 'event_type' not in log_entry:
                            log_entry['event_type'] = 'entry'
                        if 'tracking_id' not in log_entry:
                            log_entry['tracking_id'] = None
                            
                        result = insert_log(db, log_entry)
                        
                        # Update daily summary with just the date
                        update_daily_summary(db, log_entry['timestamp'].date())
                        
                        print(f"[INFO] Logged: {log_entry['person_name']} at {log_entry['camera_name']}")
                        
                except Exception as db_error:
                    print(f"[ERROR] Database insert failed: {db_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue processing other entries
                    continue
                
            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Database worker error: {e}")
                logging.error(f"Database worker error: {e}")
                time.sleep(1)
        
        print("[INFO] Database worker stopped")
    
    def start(self):
        """Start the multi-camera recognition system"""
        if self.running:
            print("[WARNING] System is already running")
            return
        
        self.running = True
        
        # Initialize database
        try:
            init_db()
            print("[INFO] Database initialized")
        except Exception as e:
            print(f"[ERROR] Failed to initialize database: {e}")
            return
        
        # Start database worker
        db_thread = threading.Thread(target=self.database_worker, daemon=True)
        db_thread.start()
        self.threads.append(db_thread)
        
        # Start camera threads
        for camera_info in self.config['cameras']:
            if camera_info.get('enabled', True):
                thread = threading.Thread(
                    target=self.process_camera, 
                    args=(camera_info,), 
                    daemon=True
                )
                thread.start()
                self.threads.append(thread)
        
        print(f"[INFO] Started {len([c for c in self.config['cameras'] if c.get('enabled', True)])} camera(s)")
    
    def stop(self):
        """Stop the multi-camera recognition system"""
        if not self.running:
            print("[WARNING] System is not running")
            return
        
        print("[INFO] Stopping system...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5)
        
        self.threads.clear()
        print("[INFO] System stopped")
    
    def get_camera_frame(self, camera_id):
        """Get the latest frame from a camera"""
        with self.data_lock:
            if camera_id in self.camera_data:
                return self.camera_data[camera_id]['frame'].copy()
        return None
    
    def get_system_status(self):
        """Get system status information"""
        with self.data_lock:
            status = {
                'running': self.running,
                'cameras': {},
                'total_known_faces': len(self.images_names),
                'active_cameras': len(self.camera_data)
            }
            
            for camera_id, data in self.camera_data.items():
                status['cameras'][camera_id] = {
                    'detections': data['detections'],
                    'last_update': data['timestamp']
                }
            
            return status

def main():
    """Main function to run the multi-camera face recognition system"""
    # Create system instance
    system = MultiCameraFaceRecognition()
    
    try:
        # Start the system
        system.start()
        
        print("[INFO] System running. Press Ctrl+C to stop...")
        
        # Keep main thread alive
        while system.running:
            time.sleep(1)
            
            # Print status every 30 seconds
            if int(time.time()) % 30 == 0:
                status = system.get_system_status()
                print(f"[STATUS] Active cameras: {status['active_cameras']}, Known faces: {status['total_known_faces']}")
    
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    
    finally:
        system.stop()

if __name__ == "__main__":
    main()