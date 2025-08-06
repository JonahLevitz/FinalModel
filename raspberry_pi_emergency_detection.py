#!/usr/bin/env python3
"""
Raspberry Pi Emergency Detection System
Optimized for Raspberry Pi Zero with Camera Module
"""

import cv2
import numpy as np
import time
import datetime
import json
import threading
from pathlib import Path
from ultralytics import YOLO
import os
import logging
from picamera2 import Picamera2
from picamera2.encoders import JpegEncoder
from picamera2.outputs import FileOutput
import io

# Set up logging for Raspberry Pi
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/home/pi/emergency_detection.log'),
        logging.StreamHandler()
    ]
)

class RaspberryPiEmergencyDetector:
    """Emergency detection system optimized for Raspberry Pi"""
    
    def __init__(self, model_path: str = "models/emergency_detection.pt"):
        """Initialize the Raspberry Pi detector"""
        self.model_path = model_path
        self.confidence_thresholds = {
            'car_crash': 0.3,
            'fire': 0.6,
            'person_fainted': 0.6
        }
        
        # Emergency class mapping
        self.emergency_classes = {
            0: 'car_crash',
            2: 'fire', 
            4: 'person_fainted'
        }
        
        # Detection history
        self.detections = []
        self.is_running = False
        
        # Create directories
        self.detections_dir = Path("/home/pi/emergency_detections")
        self.detections_dir.mkdir(exist_ok=True)
        
        # Load the trained model
        logging.info("🤖 Loading trained emergency detection model...")
        try:
            self.model = YOLO(model_path)
            logging.info(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            raise
        
        # Initialize Pi Camera
        self.setup_camera()
        
        logging.info("🚨 Raspberry Pi Emergency Detection System")
        logging.info("=" * 50)
        logging.info("Detecting: Car Crashes, Fires, Fainted People")
        logging.info("Optimized for Raspberry Pi Zero with Camera Module")
    
    def setup_camera(self):
        """Initialize Pi Camera 2"""
        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={"size": (640, 480)},
                encode="main"
            )
            self.picam2.configure(config)
            self.picam2.start()
            logging.info("📷 Pi Camera initialized successfully")
        except Exception as e:
            logging.error(f"❌ Error initializing Pi Camera: {e}")
            raise
    
    def capture_frame(self):
        """Capture frame from Pi Camera"""
        try:
            # Capture frame
            frame = self.picam2.capture_array()
            
            # Convert from RGB to BGR (OpenCV format)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            return frame
        except Exception as e:
            logging.error(f"❌ Error capturing frame: {e}")
            return None
    
    def process_frame(self, frame):
        """Process frame with emergency detection"""
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
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
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                            
                            # Create detection object
                            detection = {
                                'timestamp': datetime.datetime.now().isoformat(),
                                'time': datetime.datetime.now().strftime("%H:%M:%S"),
                                'date': datetime.datetime.now().strftime("%Y-%m-%d"),
                                'type': event_type,
                                'confidence': confidence,
                                'bbox': bbox,
                                'location': 'Raspberry Pi Camera'
                            }
                            
                            detections.append(detection)
                            self.detections.append(detection)
                            
                            logging.info(f"🚨 EMERGENCY DETECTED: {event_type} (confidence: {confidence:.2f})")
                            
                            # Save detection image
                            self.save_detection(frame, detection)
                            
                            # Send alert (can be extended for SMS/email)
                            self.send_alert(detection)
            
            return detections
            
        except Exception as e:
            logging.error(f"❌ Error processing frame: {e}")
            return []
    
    def save_detection(self, frame, detection):
        """Save detection image"""
        try:
            timestamp = detection['timestamp'].replace(':', '-').replace('.', '-')
            filename = f"pi_detection_{timestamp}_{detection['type']}_{detection['confidence']:.2f}.jpg"
            filepath = self.detections_dir / filename
            
            # Draw bounding box on frame
            x1, y1, x2, y2 = map(int, detection['bbox'])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
            
            # Add text
            label = f"{detection['type']}: {detection['confidence']:.2f}"
            cv2.putText(frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imwrite(str(filepath), frame)
            logging.info(f"💾 Detection saved: {filename}")
            
        except Exception as e:
            logging.error(f"❌ Error saving detection: {e}")
    
    def send_alert(self, detection):
        """Send emergency alert (placeholder for future implementation)"""
        try:
            alert_message = f"🚨 EMERGENCY DETECTED: {detection['type']} at {detection['time']}"
            logging.warning(alert_message)
            
            # Future implementations:
            # - Send SMS via Twilio
            # - Send email notification
            # - Trigger GPIO alarm
            # - Send to cloud dashboard
            
        except Exception as e:
            logging.error(f"❌ Error sending alert: {e}")
    
    def get_stats(self):
        """Get detection statistics"""
        if not self.detections:
            return {
                'total_detections': 0,
                'car_crashes': 0,
                'fires': 0,
                'fainted_people': 0,
                'last_detection': None
            }
        
        stats = {
            'total_detections': len(self.detections),
            'car_crashes': len([d for d in self.detections if d['type'] == 'car_crash']),
            'fires': len([d for d in self.detections if d['type'] == 'fire']),
            'fainted_people': len([d for d in self.detections if d['type'] == 'person_fainted']),
            'last_detection': self.detections[-1] if self.detections else None
        }
        
        return stats
    
    def start_monitoring(self):
        """Start continuous monitoring"""
        def monitoring_loop():
            logging.info("📹 Starting Raspberry Pi emergency monitoring...")
            last_process_time = 0
            process_interval = 3.0  # Process every 3 seconds for Pi optimization
            
            while self.is_running:
                try:
                    # Capture frame
                    frame = self.capture_frame()
                    
                    if frame is not None:
                        current_time = time.time()
                        
                        # Process every 3 seconds
                        if current_time - last_process_time >= process_interval:
                            detections = self.process_frame(frame)
                            last_process_time = current_time
                            
                            if detections:
                                logging.info(f"📊 Processed frame: {len(detections)} detections")
                    
                    time.sleep(0.5)  # Small delay to prevent overheating
                    
                except Exception as e:
                    logging.error(f"❌ Error in monitoring loop: {e}")
                    time.sleep(1)
            
            logging.info("📹 Raspberry Pi monitoring stopped")
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(target=monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logging.info("✅ Monitoring started")
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.is_running = False
        if hasattr(self, 'picam2'):
            self.picam2.close()
        logging.info("🛑 Monitoring stopped")
    
    def save_detection_log(self):
        """Save detection log to JSON file"""
        try:
            log_file = self.detections_dir / "detection_log.json"
            with open(log_file, 'w') as f:
                json.dump(self.detections, f, indent=2, default=str)
            logging.info(f"📝 Detection log saved to {log_file}")
        except Exception as e:
            logging.error(f"❌ Error saving detection log: {e}")

def main():
    """Main function for Raspberry Pi deployment"""
    try:
        # Initialize detector
        detector = RaspberryPiEmergencyDetector()
        
        # Start monitoring
        detector.start_monitoring()
        
        # Keep running until interrupted
        try:
            while True:
                time.sleep(10)
                
                # Print stats every 10 seconds
                stats = detector.get_stats()
                logging.info(f"📊 Stats: {stats['total_detections']} total detections")
                
        except KeyboardInterrupt:
            logging.info("🛑 Interrupted by user")
        
        finally:
            detector.stop_monitoring()
            detector.save_detection_log()
            
            # Print final statistics
            stats = detector.get_stats()
            logging.info(f"\n📊 Final Statistics:")
            logging.info(f"   Total detections: {stats['total_detections']}")
            logging.info(f"   Car crashes: {stats['car_crashes']}")
            logging.info(f"   Fires: {stats['fires']}")
            logging.info(f"   Fainted people: {stats['fainted_people']}")
            logging.info(f"   Detection images saved to: {detector.detections_dir}")
    
    except Exception as e:
        logging.error(f"❌ Fatal error: {e}")
        raise

if __name__ == "__main__":
    main() 