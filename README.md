# SeekBot: Advanced Campus Surveillance System

![image](https://github.com/user-attachments/assets/f907d090-6b58-4c2e-8ba1-003e72312b0a)


## Overview

SeekBot is a state-of-the-art campus surveillance system that leverages computer vision and machine learning technologies to enhance security in educational institutions. The application provides real-time monitoring, object detection, threat identification, and anomaly detection to protect students, faculty, and infrastructure.

## Features

### 🔍 Real-Time Monitoring
- Advanced object detection and tracking for immediate threat identification
- Live video feed processing with minimal latency
- Multi-camera support for comprehensive coverage

### 🚨 Anomaly Detection
- Detection of unusual behavior patterns 
- Identification of sudden movements like running
- Recognition of crowd formation and potential gatherings

### 📊 Predictive Analytics
- Historical pattern analysis for proactive threat prevention
- Data-driven insights for security resource allocation
- Trend visualization for security planning

### ⚡ Instant Alerts
- Real-time notification system for security personnel
- Multi-channel alerting (dashboard, email, mobile)
- Configurable alert thresholds

## Technology Stack

### Frontend
- React 19 for modern UI components
- React Router for navigation
- Vite for fast development and optimized builds
- CSS for custom styling

### Backend
- Python for machine learning and computer vision processing
- Object detection algorithms (YOLO/SSD)
- Video processing libraries

## Installation

### Prerequisites
- Node.js (v18.0.0 or later)
- npm or yarn
- Python 3.8+ (for backend services)

### Setup Instructions

1. **Clone the repository**
   ```bash
   git https://github.com/pranavacchu/Campus-Surveillance-System-using-YOLOv11-and-ByteTrack-for-Real-Time-Object-Tracking.git
   cd my-react-app
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Set up the backend**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

4. **Start the development server**
   ```bash
   npm run dev
   ```

5. **Build for production**
   ```bash
   npm run build
   ```

## Project Structure

```
my-react-app/
├── public/               # Static files
├── src/                  # Source files
│   ├── assets/           # Images and other assets
│   ├── backend/          # Backend integration code
│   ├── App.jsx           # Main application component
│   ├── Dashboard.jsx     # Security monitoring dashboard
│   ├── Dataset.jsx       # Dataset management
│   ├── Homepage.jsx      # Landing page
│   └── main.jsx          # Application entry point
├── index.html            # HTML entry point
└── vite.config.js        # Vite configuration
```

## Usage

### Dashboard
The Dashboard provides a comprehensive view of all connected cameras and monitoring systems. Security personnel can:
- View live feeds from multiple cameras
- Receive real-time alerts about detected anomalies
- Access historical data and analytics

### Dataset Management
The Dataset section allows administrators to:
- Manage training data for the machine learning models
- View detection accuracy statistics
- Fine-tune the system parameters

## Deployment

### Production Build
```bash
npm run build
```

The built assets will be available in the `dist` directory.



## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [React](https://reactjs.org/)
- [Vite](https://vitejs.dev/)
- [YOLO](https://pjreddie.com/darknet/yolo/) for object detection algorithms
- Faculty advisors and mentors who provided guidance throughout the project
