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
from app.db.crud import insert_log, init_db
from app.db.database import get_db
from .shared_state import latest_frames, frame_lock


# Logging setup
logging.basicConfig(
    filename='attendance.log', 
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class MultiCameraFaceRecognition:
    def __init__(self, config_path="camera_config.yaml"):
        self.config = self.load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        
        # Initialize models
        self._init_models()
        
        # Load face database
        self.images_names, self.images_embs = read_features("./datasets/face_features/feature")
        
        # Threading and data structures
        self.camera_data = {}
        self.data_lock = threading.Lock()
        self.log_queue = Queue()
        self.id_face_mapping = defaultdict(dict)
        self.last_logged_time = {}
        self.recognition_history = defaultdict(lambda: deque(maxlen=3))
        
        # Create snapshots directory
        os.makedirs("snapshots", exist_ok=True)
        
        # Get settings from config
        self.confidence_threshold = self.config['recognition']['confidence_threshold']
        self.logging_interval = self.config['recognition']['logging_interval']
        self.frame_skip = self.config['recognition']['frame_skip']
        self.max_width = self.config['performance']['max_resolution_width']
        
        # Initialize global frames dict
        global latest_frames
        for cam in self.config['cameras']:
            if cam.get('enabled', True):
                latest_frames[cam['camera_id']] = np.zeros((480, 640, 3), dtype=np.uint8)
    
    def load_config(self, config_path):
        """Load camera configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _init_models(self):
        """Initialize face detection and recognition models"""
        self.detector = SCRFD(model_file="face_detection/scrfd/weights/scrfd_2.5g_bnkps.onnx")
        self.recognizer = iresnet_inference(
            "r100", 
            "face_recognition/arcface/weights/arcface_r100.pth", 
            self.device
        )
        self.recognizer.eval()
        
        self.face_preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((112, 112)),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    
    @torch.no_grad()
    def get_feature(self, face_image):
        """Extract features from face image"""
        try:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            face_tensor = self.face_preprocess(face_rgb).unsqueeze(0).to(self.device)
            emb = self.recognizer(face_tensor).cpu().numpy()
            return emb / np.linalg.norm(emb)
        except Exception as e:
            print(f"[WARNING] Feature extraction failed: {e}")
            return None
    
    def recognize_face(self, face_image):
        """Recognize a single face"""
        emb = self.get_feature(face_image)
        if emb is None:
            return 0.0, "UN_KNOWN"
        
        score, id_min = compare_encodings(emb, self.images_embs)
        name = self.images_names[id_min]
        return score[0], name
    
    def mapping_bbox(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        inter = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return inter / (area1 + area2 - inter)
    
    def should_log_person(self, person_name, tracking_id, camera_id):
        """Check if person should be logged based on time interval"""
        now = time.time()
        key = f"{camera_id}_{person_name}_{tracking_id}"
        
        if key not in self.last_logged_time or now - self.last_logged_time[key] >= self.logging_interval:
            self.last_logged_time[key] = now
            return True
        return False
    
    def is_stable_recognition(self, tracking_id, name, camera_id):
        """Check if recognition is stable over multiple frames"""
        key = f"{camera_id}_{tracking_id}"
        self.recognition_history[key].append(name)
        recent = list(self.recognition_history[key])
        
        if len(recent) < 2:
            return False
        
        name_count = recent.count(name)
        return name_count / len(recent) >= 0.6
    
    def log_attendance(self, person_name, tracking_id, score, camera_id, event_type):
        """Log attendance to database"""
        IST = pytz.timezone('Asia/Kolkata')
        timestamp = datetime.now(IST)
        confidence = f"{score:.2f}" if score else "N/A"
        
        snapshot_path = f"snapshots/{camera_id}_{person_name}_{int(time.time())}.jpg"
        
        logging.info(f"ATTENDANCE | {timestamp} | {person_name} | ID:{tracking_id} | "
                    f"CONFIDENCE:{confidence} | CAMERA:{camera_id} | EVENT:{event_type}")
        
        try:
            db = next(get_db())  # Get a database session
            success = insert_log(
                db=db,
                person_name=person_name,
                tracking_id=tracking_id,
                confidence_score=float(score) if score else None,
                camera_id=camera_id,
                event_type=event_type,
                snapshot_path=snapshot_path,
                timestamp=timestamp
            )
            print(f"[DEBUG] {'Successfully' if success else 'Failed to'} log for {person_name}")
        except Exception as e:
            print(f"[ERROR] Database logging failed: {e}")

    def camera_worker(self, camera_config):
        """Worker thread for each camera - NO GUI DISPLAY"""
        camera_id = camera_config['camera_id']
        camera_url = camera_config['url']
        event_type = camera_config['event_type']
        
        print(f"[INFO] Starting camera worker for {camera_id}")
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_url)
        if not cap.isOpened():
            print(f"[ERROR] Cannot connect to camera '{camera_id}' at {camera_url}")
            return
        
        # Optimize camera settings
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Load tracking config
        with open("config/tracking_config.yaml", "r") as f:
            track_config = yaml.safe_load(f)
        
        tracker = BYTETracker(args=track_config, frame_rate=30)
        frame_id = 0
        frame_count = 0
        fps = 0
        start_time = time.time_ns()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"[WARNING] Failed to read frame from camera {camera_id}")
                time.sleep(0.1)
                continue
            
            # Resize if too large
            height, width = frame.shape[:2]
            if width > self.max_width:
                scale = self.max_width / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            # Skip frames for performance
            frame_id += 1
            if frame_id % self.frame_skip != 0:
                continue
            
            # Detection and tracking
            outputs, img_info, bboxes, landmarks = self.detector.detect_tracking(image=frame)
            
            tracking_bboxes = []
            tracking_ids = []
            tracking_tlwhs = []
            
            if outputs is not None:
                online_targets = tracker.update(
                    outputs, 
                    [img_info["height"], img_info["width"]], 
                    (128, 128)
                )
                
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if (tlwh[2] * tlwh[3] > track_config["min_box_area"] and 
                        tlwh[2] / tlwh[3] <= track_config["aspect_ratio_thresh"]):
                        x1, y1, w, h = tlwh
                        tracking_bboxes.append([int(x1), int(y1), int(x1 + w), int(y1 + h)])
                        tracking_tlwhs.append(tlwh)
                        tracking_ids.append(tid)
                
                # Create tracking visualization
                tracking_image = plot_tracking(
                    img_info["raw_img"], 
                    tracking_tlwhs, 
                    tracking_ids,
                    names=self.id_face_mapping[camera_id], 
                    frame_id=frame_id, 
                    fps=fps
                )
                
                # Update global frame for FastAPI streaming
                with frame_lock:
                    latest_frames[camera_id] = tracking_image.copy()
            else:
                tracking_image = img_info["raw_img"]
                with frame_lock:
                    latest_frames[camera_id] = tracking_image.copy()
            
            # Update shared data
            with self.data_lock:
                self.camera_data[camera_id] = {
                    "raw_image": img_info["raw_img"].copy(),
                    "detection_bboxes": bboxes if bboxes is not None else [],
                    "detection_landmarks": landmarks if landmarks is not None else [],
                    "tracking_ids": tracking_ids,
                    "tracking_bboxes": tracking_bboxes,
                    "event_type": event_type
                }
            
            # Calculate FPS
            frame_count += 1
            if frame_count >= 30:
                fps = 1e9 * frame_count / (time.time_ns() - start_time)
                frame_count = 0
                start_time = time.time_ns()
        
        cap.release()
        print(f"[INFO] Camera worker {camera_id} stopped")
    
    def recognition_worker(self):
        """Recognition worker for all cameras"""
        print("[INFO] Starting recognition worker")
        
        while True:
            time.sleep(self.config['performance']['recognition_interval'])
            
            with self.data_lock:
                camera_data_copy = self.camera_data.copy()
            
            for camera_id, data in camera_data_copy.items():
                if data.get("raw_image") is None:
                    continue
                
                raw_image = data["raw_image"]
                detection_landmarks = data.get("detection_landmarks", [])
                detection_bboxes = data.get("detection_bboxes", [])
                tracking_ids = data.get("tracking_ids", [])
                tracking_bboxes = data.get("tracking_bboxes", [])
                
                # Convert bboxes to lists but keep landmarks as numpy arrays
                if hasattr(detection_bboxes, 'tolist'):
                    detection_bboxes = detection_bboxes.tolist() if detection_bboxes is not None else []
                if hasattr(tracking_bboxes, 'tolist'):
                    tracking_bboxes = tracking_bboxes.tolist() if tracking_bboxes is not None else []
                
                if detection_landmarks is None:
                    detection_landmarks = []
                
                event_type = data.get("event_type", "login")
                
                if len(tracking_bboxes) == 0 or len(detection_bboxes) == 0:
                    continue
                
                # Process each tracked face
                for i, track_box in enumerate(tracking_bboxes):
                    best_match_idx = -1
                    best_iou = 0
                    
                    # Find best matching detection
                    for j, detect_box in enumerate(detection_bboxes):
                        iou = self.mapping_bbox(track_box, detect_box)
                        if iou > best_iou and iou > 0.7:
                            best_iou = iou
                            best_match_idx = j
                    
                    if best_match_idx == -1:
                        continue
                    
                    try:
                        # Extract landmark
                        if len(detection_landmarks) <= best_match_idx:
                            continue
                            
                        landmark = detection_landmarks[best_match_idx]
                        
                        if isinstance(landmark, list):
                            landmark = np.array(landmark)
                        
                        if landmark is None or len(landmark) == 0:
                            continue
                        
                        if landmark.shape != (5, 2):
                            print(f"[WARNING] Invalid landmark shape: {landmark.shape}, expected (5, 2)")
                            continue
                            
                        face = norm_crop(img=raw_image, landmark=landmark)
                        if face is None or face.size == 0:
                            continue
                        
                        score, name = self.recognize_face(face)
                        tracking_id = tracking_ids[i]
                        
                        print(f"[DEBUG] Camera {camera_id}: Recognized {name} with score {score:.3f}")
                        
                        if score < self.confidence_threshold:
                            name = "UN_KNOWN"
                        
                        if name != "UN_KNOWN" and not self.is_stable_recognition(tracking_id, name, camera_id):
                            print(f"[DEBUG] Unstable recognition for {name}, waiting for stability")
                            continue
                        
                        # Update display mapping
                        display_name = name if name == "UN_KNOWN" else f"{name}:{score:.2f}"
                        self.id_face_mapping[camera_id][tracking_id] = display_name
                        
                        # Log attendance
                        if self.should_log_person(name, tracking_id, camera_id):
                            snapshot_path = f"snapshots/{camera_id}_{name}_{int(time.time())}.jpg"
                            cv2.imwrite(snapshot_path, raw_image)
                            print(f"[INFO] Logging attendance: {name} from camera {camera_id}")
                            
                            self.log_attendance(
                                name, tracking_id, 
                                score if name != "UN_KNOWN" else None,
                                camera_id, event_type
                            )
                    
                    except Exception as e:
                        print(f"[WARNING] Recognition failed for camera {camera_id}: {e}")
                        continue

    def log_worker(self):
        """Database logging worker"""
        print("[INFO] Starting log worker")

        while True:
            try:
                log_entry = self.log_queue.get(timeout=1)
                print(f"[DEBUG] Processing log: {log_entry[0]} from camera {log_entry[3]}")
                insert_log(*log_entry)
                self.log_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Database logging failed: {e}")
                import traceback
                traceback.print_exc()
    
    def start(self):
        """Start the multi-camera recognition system"""
        print("[INFO] Initializing multi-camera face recognition system...")
        
        # Initialize database
        init_db()
        
        # Get enabled cameras
        enabled_cameras = [cam for cam in self.config['cameras'] if cam.get('enabled', True)]
        
        if not enabled_cameras:
            print("[ERROR] No enabled cameras found in configuration")
            return
        
        print(f"[INFO] Found {len(enabled_cameras)} enabled cameras")
        
        # Start worker threads
        threads = []
        
        # Start log worker
        log_thread = threading.Thread(target=self.log_worker, daemon=True)
        log_thread.start()
        threads.append(log_thread)
        
        # Start recognition worker
        recognition_thread = threading.Thread(target=self.recognition_worker, daemon=True)
        recognition_thread.start()
        threads.append(recognition_thread)
        
        # Start camera workers
        for camera_config in enabled_cameras:
            camera_thread = threading.Thread(
                target=self.camera_worker, 
                args=(camera_config,), 
                daemon=True
            )
            camera_thread.start()
            threads.append(camera_thread)
        
        print("[INFO] All workers started.")
        print("[INFO] Access camera streams at:")
        for cam in enabled_cameras:
            print(f"  - {cam['camera_id']}: http://localhost:8000/stream/{cam['camera_id']}")
        
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down...")

def main():
    """Main function"""
    recognition_system = MultiCameraFaceRecognition("camera_config.yaml")
    recognition_system.start()

# if __name__ == "__main__":
#     main()