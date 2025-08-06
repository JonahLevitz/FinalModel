#!/bin/bash

# Raspberry Pi Emergency Detection System Setup
# This script sets up the environment for running the emergency detection system

echo "🚨 Raspberry Pi Emergency Detection System Setup"
echo "================================================"

# Update system
echo "📦 Updating system packages..."
sudo apt update && sudo apt upgrade -y

# Install required packages
echo "📦 Installing required packages..."
sudo apt install -y python3-pip python3-venv git cmake build-essential

# Install Pi Camera dependencies
echo "📷 Installing Pi Camera dependencies..."
sudo apt install -y python3-picamera2 python3-opencv

# Install PyTorch for ARM (optimized for Pi)
echo "🤖 Installing PyTorch for Raspberry Pi..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other Python dependencies
echo "📦 Installing Python dependencies..."
pip3 install ultralytics opencv-python numpy pillow

# Create project directory
echo "📁 Setting up project directory..."
mkdir -p /home/pi/emergency_detection
cd /home/pi/emergency_detection

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install dependencies in virtual environment
echo "📦 Installing dependencies in virtual environment..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics opencv-python numpy pillow

# Create directories
echo "📁 Creating directories..."
mkdir -p models
mkdir -p emergency_detections
mkdir -p logs

# Download the trained model (you'll need to transfer this)
echo "🤖 Model setup..."
echo "   Please transfer your trained emergency_detection.pt model to /home/pi/emergency_detection/models/"

# Create systemd service for auto-start
echo "⚙️ Creating systemd service..."
sudo tee /etc/systemd/system/emergency-detection.service > /dev/null <<EOF
[Unit]
Description=Emergency Detection System
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/emergency_detection
Environment=PATH=/home/pi/emergency_detection/venv/bin
ExecStart=/home/pi/emergency_detection/venv/bin/python raspberry_pi_emergency_detection.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service
echo "🔧 Enabling systemd service..."
sudo systemctl daemon-reload
sudo systemctl enable emergency-detection.service

# Create startup script
echo "📝 Creating startup script..."
tee /home/pi/emergency_detection/start.sh > /dev/null <<EOF
#!/bin/bash
cd /home/pi/emergency_detection
source venv/bin/activate
python raspberry_pi_emergency_detection.py
EOF

chmod +x /home/pi/emergency_detection/start.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
tee /home/pi/emergency_detection/monitor.sh > /dev/null <<EOF
#!/bin/bash
echo "📊 Emergency Detection System Status"
echo "==================================="
echo "Service status:"
sudo systemctl status emergency-detection.service
echo ""
echo "Recent logs:"
tail -20 /home/pi/emergency_detection.log
echo ""
echo "Detection count:"
ls /home/pi/emergency_detections/*.jpg 2>/dev/null | wc -l
echo "detection images saved"
EOF

chmod +x /home/pi/emergency_detection/monitor.sh

# Create README for Pi
echo "📖 Creating README..."
tee /home/pi/emergency_detection/README_PI.md > /dev/null <<EOF
# Raspberry Pi Emergency Detection System

## 🚨 Quick Start

1. **Transfer your trained model:**
   - Copy your \`emergency_detection.pt\` model to \`/home/pi/emergency_detection/models/\`

2. **Start the system:**
   \`\`\`bash
   cd /home/pi/emergency_detection
   ./start.sh
   \`\`\`

3. **Monitor the system:**
   \`\`\`bash
   ./monitor.sh
   \`\`\`

4. **View logs:**
   \`\`\`bash
   tail -f /home/pi/emergency_detection.log
   \`\`\`

## 📁 File Structure

- \`raspberry_pi_emergency_detection.py\` - Main detection script
- \`models/emergency_detection.pt\` - Trained model (transfer this)
- \`emergency_detections/\` - Saved detection images
- \`logs/\` - System logs

## 🔧 System Commands

- **Start service:** \`sudo systemctl start emergency-detection\`
- **Stop service:** \`sudo systemctl stop emergency-detection\`
- **View status:** \`sudo systemctl status emergency-detection\`
- **View logs:** \`journalctl -u emergency-detection -f\`

## 📊 Features

- **Real-time monitoring** with Pi Camera
- **Automatic detection** of car crashes, fires, fainted people
- **Image saving** with timestamps and confidence scores
- **Logging** to file and system
- **Auto-restart** on failure
- **Optimized** for Raspberry Pi Zero

## 🚨 Emergency Alerts

The system will:
- Log all detections to \`/home/pi/emergency_detection.log\`
- Save detection images to \`/home/pi/emergency_detections/\`
- Print alerts to console

## 🔄 Future Enhancements

- SMS alerts via Twilio
- Email notifications
- GPIO alarm triggers
- Cloud dashboard integration
- Web interface for monitoring

## 📞 Troubleshooting

1. **Check camera:** \`vcgencmd get_camera\`
2. **Check logs:** \`tail -f /home/pi/emergency_detection.log\`
3. **Restart service:** \`sudo systemctl restart emergency-detection\`
4. **Check disk space:** \`df -h\`

## 🔋 Power Management

- The system is optimized for low power consumption
- Processing interval: 3 seconds
- Automatic sleep between captures
- Efficient memory usage
EOF

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Transfer your trained model to /home/pi/emergency_detection/models/"
echo "2. Run: cd /home/pi/emergency_detection && ./start.sh"
echo "3. Monitor with: ./monitor.sh"
echo ""
echo "📖 See README_PI.md for detailed instructions"
echo ""
echo "🚨 Emergency Detection System ready for Raspberry Pi deployment!" 