from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import time
import asyncio
from .shared_state import latest_frames, frame_lock

app = FastAPI()

def create_placeholder_frame(camera_id, message="Starting..."):
    """Create a placeholder frame when no camera feed is available"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (40, 40, 40)  # Dark gray background
    
    # Add camera ID text
    cv2.putText(frame, f"Camera: {camera_id}", (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Add status message
    cv2.putText(frame, message, (20, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 100, 255), 2)
    
    # Add timestamp
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (20, 450), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
    
    return frame

async def generate_frames(camera_id):
    """Generate video frames for streaming"""
    while True:
        try:
            # Get the latest frame with thread safety
            with frame_lock:
                frame = latest_frames.get(camera_id, None)
                if frame is None:
                    frame = create_placeholder_frame(camera_id, "Waiting for Recognition System...")
                else:
                    frame = frame.copy()  # Make a copy to avoid modifying the original
            
            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            
            # Yield the frame in the multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            
            # Control frame rate (~25fps)
            await asyncio.sleep(0.04)
            
        except Exception as e:
            print(f"Stream error for {camera_id}: {e}")
            await asyncio.sleep(1)

@app.get("/")
async def home():
    """Simple home page with camera streams"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Camera Streams</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background: #f0f2f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
            }
            .camera-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(600px, 1fr));
                gap: 20px;
            }
            .camera-card {
                background: white;
                border-radius: 8px;
                padding: 15px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            }
            .camera-title {
                font-size: 1.2em;
                margin-bottom: 10px;
                color: #333;
            }
            .stream-container {
                position: relative;
                padding-bottom: 56.25%; /* 16:9 aspect ratio */
                height: 0;
                overflow: hidden;
            }
            .stream-container img {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                object-fit: contain;
                background: #222;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Multi-Camera Face Recognition</h1>
                <p>Real-time monitoring system</p>
            </div>
            
            <div class="camera-grid">
                <div class="camera-card">
                    <div class="camera-title">Entry Camera</div>
                    <div class="stream-container">
                        <img src="/stream/entry_01" alt="Entry Camera Stream">
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Auto-reconnect if stream fails
            document.querySelectorAll('img').forEach(img => {
                img.addEventListener('error', function() {
                    console.log('Stream error, attempting to reconnect...');
                    setTimeout(() => {
                        this.src = this.src.split('?')[0] + '?t=' + new Date().getTime();
                    }, 1000);
                });
                
                // Force refresh every 2 seconds if no image loaded
                setTimeout(() => {
                    if (this.complete && this.naturalWidth === 0) {
                        this.src = this.src.split('?')[0] + '?t=' + new Date().getTime();
                    }
                }, 2000);
            });
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/stream/{camera_id}")
async def video_stream(camera_id: str):
    """Video streaming endpoint for a specific camera"""
    return StreamingResponse(
        generate_frames(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.on_event("startup")
async def startup_event():
    """Initialize the application"""
    print("FastAPI Stream Server Starting...")
    print("Access the web interface at: http://localhost:8000")
    
    # Initialize frames for all expected cameras
    with frame_lock:
        latest_frames["entry_01"] = create_placeholder_frame("entry_01")

# recognition_system = None

# @app.on_event("startup")
# async def startup_event():
#     print("FastAPI Stream Server Starting...")
#     print("Access the web interface at: http://localhost:8000")
    
#     # Start recognition system in background
#     def run_recognition():
#         global recognition_system
#         recognition_system = MultiCameraFaceRecognition("camera_config.yaml")
#         recognition_system.start()
    
#     threading.Thread(target=run_recognition, daemon=True).start()

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)