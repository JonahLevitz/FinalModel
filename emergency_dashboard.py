#!/usr/bin/env python3
"""
Emergency Detection Dashboard
Real-time web dashboard for tracking emergency detections
"""

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import time
import datetime
import json
import threading
from pathlib import Path
from ultralytics import YOLO
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emergency_detection_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

class EmergencyDashboard:
    """Dashboard for emergency detection system"""
    def __init__(self):
        self.detections = []
        self.is_running = False
        self.webcam_thread = None
        self.model = None
        self.cap = None
        
        # Emergency class mapping
        self.emergency_classes = {
            0: 'car_crash',
            2: 'fire',
            4: 'person_fainted'
        }
        
        # Detection cooldown to prevent spam
        self.last_detection_time = {}
        self.detection_cooldown = 10  # seconds between same type detections
        
        # Confidence thresholds - Increased car_crash threshold to reduce false positives
        self.confidence_thresholds = {
            'car_crash': 0.7,  # Increased from 0.3 to 0.7
            'fire': 0.6,
            'person_fainted': 0.6
        }
        
        # Load model
        self.load_model()
    
    def load_model(self):
        """Load the trained emergency detection model"""
        try:
            self.model = YOLO('models/emergency_detection.pt')
            print("✅ Model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
    
    def add_detection(self, event_type, confidence, location="Webcam Feed"):
        """Add a new detection to the dashboard"""
        detection = {
            'id': len(self.detections) + 1,
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'time': datetime.datetime.now().strftime("%H:%M:%S"),
            'date': datetime.datetime.now().strftime("%Y-%m-%d"),
            'event': event_type,
            'location': location,
            'confidence': f"{confidence:.2f}",
            'status': 'Active',
            'severity': self.get_severity(event_type, confidence)
        }
        
        self.detections.append(detection)
        
        # Emit to all connected clients
        socketio.emit('new_detection', detection)
        
        print(f"🚨 Dashboard: {event_type} detected at {detection['timestamp']}")
        return detection
    
    def get_severity(self, event_type, confidence):
        """Get severity level based on event type and confidence"""
        if confidence >= 0.8:
            return "Critical"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def process_frame(self, frame):
        """Process frame and detect emergencies"""
        try:
            results = self.model(frame, verbose=False)
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        
                        # Map to our emergency classes
                        if class_id == 0:  # car_crash
                            event_type = 'car_crash'
                            threshold = self.confidence_thresholds['car_crash']
                        elif class_id == 2:  # fire
                            event_type = 'fire'
                            threshold = self.confidence_thresholds['fire']
                        elif class_id == 4:  # person_fainted
                            event_type = 'person_fainted'
                            threshold = self.confidence_thresholds['person_fainted']
                        else:
                            continue
                        
                        if confidence >= threshold:
                            # Check cooldown to prevent spam detections
                            current_time = time.time()
                            if event_type not in self.last_detection_time or \
                               current_time - self.last_detection_time[event_type] >= self.detection_cooldown:
                                self.add_detection(event_type, confidence)
                                self.last_detection_time[event_type] = current_time
                                print(f"🔍 High confidence detection: {event_type} ({confidence:.2f})")
                            else:
                                print(f"⏰ Skipping {event_type} detection (cooldown active)")
                            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
    
    def start_webcam(self):
        """Start webcam monitoring in background thread"""
        def webcam_loop():
            self.cap = cv2.VideoCapture(0)
            
            if not self.cap.isOpened():
                print("❌ Error: Could not open webcam")
                socketio.emit('webcam_error', {'message': 'Could not open webcam'})
                return
            
            print("📹 Webcam monitoring started")
            socketio.emit('webcam_status', {'status': 'started', 'message': 'Webcam monitoring active'})
            last_process_time = 0
            process_interval = 2.0  # Process every 2 seconds
            
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("❌ Error reading frame from webcam")
                    continue
                
                current_time = time.time()
                
                # Process every 2 seconds
                if current_time - last_process_time >= process_interval:
                    print(f"🔍 Processing webcam frame at {datetime.datetime.now().strftime('%H:%M:%S')}")
                    self.process_frame(frame)
                    last_process_time = current_time
                
                time.sleep(0.1)  # Small delay
            
            if self.cap:
                self.cap.release()
            print("📹 Webcam monitoring stopped")
            socketio.emit('webcam_status', {'status': 'stopped', 'message': 'Webcam monitoring stopped'})
        
        self.is_running = True
        self.webcam_thread = threading.Thread(target=webcam_loop, daemon=True)
        self.webcam_thread.start()
    
    def stop_webcam(self):
        """Stop webcam monitoring"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        print("📹 Webcam monitoring stopped")
    
    def get_stats(self):
        """Get dashboard statistics"""
        if not self.detections:
            return {
                'total_detections': 0,
                'car_crashes': 0,
                'fires': 0,
                'fainted_people': 0,
                'critical_events': 0,
                'high_events': 0
            }
        
        stats = {
            'total_detections': len(self.detections),
            'car_crashes': len([d for d in self.detections if d['event'] == 'car_crash']),
            'fires': len([d for d in self.detections if d['event'] == 'fire']),
            'fainted_people': len([d for d in self.detections if d['event'] == 'person_fainted']),
            'critical_events': len([d for d in self.detections if d['severity'] == 'Critical']),
            'high_events': len([d for d in self.detections if d['severity'] == 'High'])
        }
        
        return stats

# Global dashboard instance
dashboard = EmergencyDashboard()

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/webcam_test')
def webcam_test():
    """Webcam testing page"""
    return render_template('webcam_test.html')

@app.route('/api/detections')
def get_detections():
    """Get all detections"""
    return jsonify(dashboard.detections)

@app.route('/api/stats')
def get_stats():
    """Get dashboard statistics"""
    return jsonify(dashboard.get_stats())

@app.route('/api/start_monitoring', methods=['POST'])
def start_monitoring():
    """Start webcam monitoring"""
    if not dashboard.is_running:
        dashboard.start_webcam()
        return jsonify({'status': 'success', 'message': 'Monitoring started'})
    return jsonify({'status': 'error', 'message': 'Monitoring already running'})

@app.route('/api/stop_monitoring', methods=['POST'])
def stop_monitoring():
    """Stop webcam monitoring"""
    dashboard.stop_webcam()
    return jsonify({'status': 'success', 'message': 'Monitoring stopped'})

@app.route('/api/add_test_detection', methods=['POST'])
def add_test_detection():
    """Add a test detection for demonstration"""
    data = request.get_json()
    event_type = data.get('event_type', 'fire')
    confidence = data.get('confidence', 0.75)
    
    detection = dashboard.add_detection(event_type, confidence, "Test Location")
    return jsonify(detection)

@app.route('/video_feed')
def video_feed():
    """Video streaming route for webcam feed"""
    def generate_frames():
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            return
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame for better performance
            frame = cv2.resize(frame, (640, 480))
            
            # Convert to JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            
            # Yield the frame in bytes
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        
        cap.release()
    
    return app.response_class(generate_frames(),
                            mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print("🔗 Client connected to dashboard")
    emit('connected', {'message': 'Connected to Emergency Dashboard'})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print("🔌 Client disconnected from dashboard")

if __name__ == '__main__':
    print("🚨 Emergency Detection Dashboard")
    print("=" * 50)
    print("📊 Dashboard Features:")
    print("  • Real-time emergency tracking")
    print("  • Time, location, and event logging")
    print("  • Live statistics and monitoring")
    print("  • Web-based interface")
    print("=" * 50)
    
    # Create templates directory if it doesn't exist
    templates_dir = Path("templates")
    templates_dir.mkdir(exist_ok=True)
    
    # Start the dashboard
    socketio.run(app, host='0.0.0.0', port=5000, debug=True) 