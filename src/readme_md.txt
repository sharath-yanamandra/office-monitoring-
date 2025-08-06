# Video Monitoring System

A flexible, AI-powered video monitoring system that allows users to select camera feeds and choose which detection models to run. Perfect for datacenter security, office monitoring, and general surveillance applications.

## ✨ Features

### 🎯 **Flexible Model Selection**
- Choose any camera feed
- Select one or multiple detection models
- Real-time processing with GPU acceleration
- No vendor lock-in - works with any RTSP camera

### 🤖 **AI Detection Models**
1. **Tailgating Detection** - Multiple people entering through single access point
2. **People Counting** - Bidirectional counting with traffic flow analysis  
3. **PPE Detection** - Safety equipment compliance monitoring
4. **Unauthorized Access** - Restricted area breach detection
5. **Intrusion Detection** - Motion-based perimeter security
6. **Loitering Detection** - Extended presence monitoring

### 🛠️ **Easy to Use**
- Simple command-line interface
- Built-in event search and filtering
- Visual frame annotations for each detection type
- Comprehensive logging and statistics

### 🚀 **Deployment Options**
- **Local Development** - Python virtual environment
- **Docker** - Containerized deployment with Docker Compose  
- **Google Cloud Platform** - Scalable cloud deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- MySQL 8.0+
- CUDA-compatible GPU (optional, for acceleration)

### Installation

1. **Clone and setup directory structure:**
```bash
mkdir video-monitoring-system
cd video-monitoring-system

# Copy all provided files according to the file structure
```

2. **Install dependencies:**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Setup database:**
```bash
# Create MySQL database
mysql -u root -p -e "CREATE DATABASE video_monitoring;"
mysql -u root -p -e "CREATE USER 'video_monitor_user'@'localhost' IDENTIFIED BY 'secure_password';"
mysql -u root -p -e "GRANT ALL PRIVILEGES ON video_monitoring.* TO 'video_monitor_user'@'localhost';"

# Initialize schema and sample data
mysql -u video_monitor_user -p video_monitoring < db_setup.sql
```

4. **Configure environment:**
```bash
cp .env.example .env
# Edit .env with your database credentials and settings
```

5. **Download AI models:**
```bash
mkdir -p models
# YOLOv11 models will be downloaded automatically on first run
# Or manually download:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov11l.pt -P models/
```

## 📖 Usage

### List Available Resources

```bash
# List cameras in database
python src/main.py list-cameras

# List available detection models  
python src/main.py list-models

# Show camera details
python src/main.py camera-details 1
```

### Start Monitoring

```bash
# Monitor with single detection model
python src/main.py monitor --camera 1 --models people_counting

# Monitor with multiple models
python src/main.py monitor --camera 1 --models tailgating people_counting ppe_detection

# Monitor with all detection models
python src/main.py monitor --camera 1 --models tailgating unauthorized_access people_counting ppe_detection intrusion_detection loitering_detection
```

### Search Events

```bash
# Search all events
python src/main.py search-events

# Search specific camera events
python src/main.py search-events --camera 1

# Search by event type and severity
python src/main.py search-events --type tailgating --severity high

# Limit results
python src/main.py search-events --camera 1 --limit 20
```

### System Maintenance

```bash
# Check system health
python src/main.py health-check

# Initialize sample data
python src/main.py init-data
```

## 🐳 Docker Deployment

### Quick Start with Docker

```bash
# Start all services
docker-compose up -d

# Initialize database
docker-compose exec app python src/main.py init-data

# Start monitoring
docker-compose exec app python src/main.py monitor --camera 1 --models people_counting

# View logs
docker-compose logs -f app
```

### Services Included

- **app** - Main video monitoring application
- **mysql** - MySQL database
- **phpmyadmin** - Database management (optional)
- **redis** - Caching layer (optional)

## ☁️ Google Cloud Platform Deployment

### Option 1: Compute Engine VM

```bash
# Create VM with GPU support
gcloud compute instances create video-monitoring-vm \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --image-family=ubuntu-2004-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB

# SSH and install system
gcloud compute ssh video-monitoring-vm --zone=us-central1-a
# Follow local installation steps
```

### Option 2: Cloud Run + Cloud SQL

```bash
# Create Cloud SQL instance
gcloud sql instances create video-monitoring-db \
  --database-version=MYSQL_8_0 \
  --tier=db-n1-standard-1 \
  --region=us-central1

# Deploy to Cloud Run
gcloud builds submit --tag gcr.io/PROJECT_ID/video-monitoring
gcloud run deploy video-monitoring \
  --image gcr.io/PROJECT_ID/video-monitoring \
  --platform managed \
  --region us-central1
```

## 🔧 Configuration

### Environment Variables (.env)

```bash
# Database
MYSQL_HOST=localhost
MYSQL_USER=video_monitor_user  
MYSQL_PASSWORD=secure_password
MYSQL_DATABASE=video_monitoring

# AI Models
DETECTION_MODEL_PATH=models/yolov11l.pt
PERSON_DETECTION_CONFIDENCE=0.5
PPE_CONFIDENCE_THRESHOLD=0.7

# Processing
BATCH_SIZE=4
GPU_MEMORY_FRACTION=0.8

# Detection Parameters
LOITERING_THRESHOLD=300
TAILGATING_TIME_WINDOW=10
```

### Detection Model Parameters

Each detection model can be customized:

- **Tailgating**: `time_window`, `max_people_per_entry`
- **People Counting**: `counting_line_threshold`, `direction_tracking`  
- **PPE Detection**: `required_ppe`, `grace_period`
- **Unauthorized Access**: `confidence_threshold`, `time_based_restrictions`
- **Intrusion Detection**: `sensitivity`, `motion_threshold`
- **Loitering**: `loitering_threshold`, `movement_threshold`

## 📊 Database Schema

### Core Tables

- **cameras** - Camera configuration and metadata
- **zones** - Detection zones with polygon coordinates  
- **events** - Detection events with timestamps and data
- **detection_sessions** - User monitoring sessions

### Sample Data

The system includes sample data for testing:
- 6 cameras with different types (entry, server room, corridor, etc.)
- Pre-configured zones for each camera type
- Example events for each detection model

## 🔍 Detection Models Details

### 1. Tailgating Detection
- **Purpose**: Detect multiple people entering through single access point
- **Algorithm**: Time window analysis + people counting
- **Visualization**: Entry timeline, people tracking
- **Events**: Triggered when >1 person enters within time window

### 2. People Counting  
- **Purpose**: Count people crossing lines or entering zones
- **Algorithm**: Kalman tracking + line crossing detection
- **Visualization**: Counting lines, traffic statistics, flow direction
- **Events**: Count updates, occupancy changes

### 3. PPE Detection
- **Purpose**: Monitor safety equipment compliance
- **Algorithm**: YOLO object detection + person-to-PPE matching
- **Visualization**: PPE bounding boxes, compliance status
- **Events**: PPE violations in restricted zones

### 4. Unauthorized Access
- **Purpose**: Detect people in restricted areas
- **Algorithm**: Zone containment + access control simulation
- **Visualization**: Authorization status, time restrictions
- **Events**: Unauthorized presence beyond tolerance period

### 5. Intrusion Detection
- **Purpose**: General security breach detection
- **Algorithm**: Background subtraction + person detection + pattern analysis
- **Visualization**: Motion areas, suspicious behavior indicators
- **Events**: Motion or person detection in secured zones

### 6. Loitering Detection
- **Purpose**: Extended presence monitoring
- **Algorithm**: Movement pattern analysis + stationary detection
- **Visualization**: Movement trails, loitering timers
- **Events**: Person stationary beyond threshold time

## 🛠️ Development

### Adding Custom Detection Models

1. Create new detection script in `src/detection_scripts/`
2. Inherit from `CameraModelBase`
3. Implement `process_frame()` method
4. Add to `DETECTOR_CLASSES` in `__init__.py`
5. Update configuration in `config.py`

### Testing

```bash
# Run health check
python src/main.py health-check

# Test with sample data
python src/main.py init-data
python src/main.py monitor --camera 1 --models people_counting
```

## 📝 Logging

### Log Files
- `logs/main.log` - Application main log
- `logs/database.log` - Database operations  
- `logs/camera_*.log` - Per-camera logs
- `logs/*_performance.log` - Performance metrics
- `logs/*_audit.log` - Audit trail

### Log Levels
- **DEBUG** - Detailed debugging information
- **INFO** - General information messages  
- **WARNING** - Warning messages
- **ERROR** - Error messages
- **CRITICAL** - Critical system errors

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

### Common Issues

1. **Database Connection Issues**
   - Check MySQL service status
   - Verify credentials in `.env`
   - Ensure database exists

2. **Model Loading Issues**
   - Verify model files in `models/` directory
   - Check GPU availability with `nvidia-smi`
   - Install PyTorch with CUDA support

3. **Camera Connection Issues**  
   - Test RTSP URL with VLC or ffmpeg
   - Check network connectivity
   - Verify camera credentials

### Getting Help

- 📖 Check the deployment guide for detailed setup instructions
- 🔧 Run `python src/main.py health-check` for system diagnostics
- 📊 Use `python src/main.py list-cameras` to verify database setup
- 🐛 Check logs in `logs/` directory for error details

---

**Built with ❤️ for flexible video monitoring**