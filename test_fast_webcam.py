#!/usr/bin/env python3
"""
Fast Webcam Test for Emergency Detection System
Optimized for real-time performance
"""

import cv2
import numpy as np
import time
import datetime
from pathlib import Path
import json
from ultralytics import YOLO

class FastEmergencyDetector:
    """Fast emergency detection system using trained model"""
    
    def __init__(self, model_path: str = "models/emergency_detection.pt"):
        """Initialize the fast detector"""
        self.model_path = model_path
        self.confidence_threshold = 0.6  # Higher threshold for fewer false positives
        self.process_interval = 2.0  # Process every 2 seconds instead of every frame
        self.last_process_time = 0
        
        # Emergency class mapping - Only the 3 main classes we want
        self.emergency_classes = [
            'car_crash', 'fire', 'person_fainted'
        ]
        
        # Emergency class colors
        self.emergency_colors = {
            'car_crash': (0, 0, 255),        # Red
            'fire': (0, 0, 255),             # Red
            'person_fainted': (255, 0, 255), # Purple
        }
        
        # Detection history
        self.detection_history = []
        self.current_detections = []  # Store current detections
        
        # Create detections directory
        self.detections_dir = Path("detections")
        self.detections_dir.mkdir(exist_ok=True)
        
        # Load the trained model
        print("🤖 Loading trained emergency detection model...")
        try:
            self.model = YOLO(model_path)
            print(f"✅ Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        print("🚁 Fast Emergency Detection System")
        print("=" * 50)
        print("Detecting: Car Crashes, Fires, Fainted People")
        print("Press 'q' to quit, 's' to save frame, 'p' to process now")
        print("⚡ Processing every 2 seconds for better performance")
    
    def process_frame(self, frame):
        """Process frame with real model (optimized)"""
        current_time = time.time()
        
        # Only process every 2 seconds to reduce lag
        if current_time - self.last_process_time < self.process_interval:
            # Return previous detections
            return self.current_detections
        
        try:
            # Run inference
            results = self.model(frame, verbose=False)
            
            detections = []
            
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes
                    
                    for box in boxes:
                        confidence = float(box.conf[0])
                        
                        if confidence >= self.confidence_threshold:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            bbox = [float(x1), float(y1), float(x2), float(y2)]
                            
                            # Get class label - Map to our 3 main classes
                            class_id = int(box.cls[0])
                            
                            # Map model classes to our 3 main classes
                            if class_id == 0:  # car_crash
                                label = 'car_crash'
                                # Lower threshold for car_crash since it has lower confidence
                                if confidence < 0.3:
                                    continue
                            elif class_id == 2:  # fire
                                label = 'fire'
                            elif class_id == 4:  # person_fainted
                                label = 'person_fainted'
                            else:
                                # Skip other classes
                                continue
                            
                            # Create detection object
                            detection = {
                                'timestamp': datetime.datetime.now().isoformat(),
                                'type': label,
                                'confidence': confidence,
                                'bbox': bbox
                            }
                            
                            detections.append(detection)
                            
                            # Add to history
                            self.detection_history.append(detection)
                            
                            print(f"🚨 REAL EMERGENCY DETECTED: {label} (confidence: {confidence:.2f})")
                            
                            # Save detection image
                            self.save_detection(frame, detection)
            
            # Update current detections and timestamp
            self.current_detections = detections
            self.last_process_time = current_time
            
            return detections
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            return self.current_detections
    
    def draw_detections(self, frame):
        """Draw current detections on frame"""
        for detection in self.current_detections:
            x1, y1, x2, y2 = map(int, detection['bbox'])
            label = detection['type']
            confidence = detection['confidence']
            
            # Get color
            color = self.emergency_colors.get(label, (0, 255, 0))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw label
            label_text = f"{label}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1-30), (x1+len(label_text)*12, y1), color, -1)
            cv2.putText(frame, label_text, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def save_detection(self, frame, detection):
        """Save detection image"""
        timestamp = detection['timestamp'].replace(':', '-').replace('.', '-')
        filename = f"fast_detection_{timestamp}_{detection['type']}_{detection['confidence']:.2f}.jpg"
        filepath = self.detections_dir / filename
        
        cv2.imwrite(str(filepath), frame)
        print(f"💾 Fast detection saved: {filename}")
    
    def draw_stats(self, frame):
        """Draw detection statistics on frame"""
        # Draw detection count
        detection_count = len(self.detection_history)
        cv2.putText(frame, f"Total Detections: {detection_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw current detections count
        current_count = len(self.current_detections)
        cv2.putText(frame, f"Current: {current_count}", (10, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw recent detections
        recent_detections = self.detection_history[-3:]  # Last 3 detections
        y_offset = 85
        for i, detection in enumerate(recent_detections):
            text = f"{detection['type']}: {detection['confidence']:.2f}"
            cv2.putText(frame, text, (10, y_offset + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw processing status
        time_since_last = time.time() - self.last_process_time
        status = f"Next process in: {max(0, self.process_interval - time_since_last):.1f}s"
        cv2.putText(frame, status, (10, frame.shape[0]-50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save, 'p' to process now", 
                   (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def run_webcam_test(self):
        """Run the fast webcam test"""
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Error: Could not open webcam")
            return
        
        print("📹 Webcam started successfully")
        print("🎯 Fast emergency detection system is running...")
        print("⚡ Processing every 2 seconds for smooth performance")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("❌ Error: Could not read frame from webcam")
                    break
                
                # Process frame (only every 2 seconds)
                detections = self.process_frame(frame)
                
                # Draw current detections
                self.draw_detections(frame)
                
                # Draw statistics
                self.draw_stats(frame)
                
                # Display frame
                cv2.imshow('Fast Emergency Detection System', frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("🛑 Quitting...")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"webcam_frame_{timestamp}.jpg"
                    filepath = self.detections_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    print(f"💾 Frame saved: {filename}")
                elif key == ord('p'):
                    # Force process now
                    self.last_process_time = 0
                    print("🔍 Forcing immediate processing...")
        
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Print final statistics
            print(f"\n📊 Fast Test Statistics:")
            print(f"   Total detections: {len(self.detection_history)}")
            
            if self.detection_history:
                print(f"   Detection types:")
                type_counts = {}
                for detection in self.detection_history:
                    emergency_type = detection['type']
                    type_counts[emergency_type] = type_counts.get(emergency_type, 0) + 1
                
                for emergency_type, count in type_counts.items():
                    print(f"     {emergency_type}: {count}")
            
            print(f"   Detection images saved to: {self.detections_dir}")

def main():
    """Main function"""
    detector = FastEmergencyDetector()
    detector.run_webcam_test()

if __name__ == "__main__":
    main() 