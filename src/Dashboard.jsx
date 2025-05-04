import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dashboard.css';
import logoImage from './camera_logo_black.jpg';
import backgroundImage from './objects_ghibli.png';

const Dashboard = () => {
  const navigate = useNavigate();
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [detections, setDetections] = useState([]);
  const videoRef = useRef(null);
  const wsRef = useRef(null);

  const handleBack = () => {
    navigate('/');
  };

  const toggleWebcam = () => {
    if (isWebcamActive) {
      if (wsRef.current) {
        wsRef.current.close();
      }
      setIsWebcamActive(false);
    } else {
      startWebcam();
    }
  };

  const startWebcam = () => {
    setIsWebcamActive(true);
    wsRef.current = new WebSocket('ws://localhost:8000/ws');

    wsRef.current.onmessage = (event) => {
      const data = JSON.parse(event.data);
      const img = new Image();
      img.src = `data:image/jpeg;base64,${data.frame}`;
      if (videoRef.current) {
        videoRef.current.src = img.src;
      }
      setDetections(data.detections);
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
      setIsWebcamActive(false);
    };

    wsRef.current.onclose = () => {
      setIsWebcamActive(false);
    };
  };

  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return (
    <div className="dashboard">
      <nav className="navbar">
        <div className="nav-left">
          <button className="back-button" onClick={handleBack}>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M15 18L9 12L15 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <div className="nav-brand">
            <img src={logoImage} alt="Logo" className="nav-logo-img" />
            <span className="nav-logo-text">SeekBot</span>
          </div>
        </div>
        <div className="nav-right">
          <div className="nav-links">
            <a href="#" onClick={(e) => { e.preventDefault(); navigate('/dataset'); }}>Datasets</a>
            <a href="#">Features</a>
            <a href="#">Pricing</a>
            <a href="#">Customers</a>
            <a href="#">Contact Us</a>
          </div>
        </div>
      </nav>

      <main className="dashboard-content">
        <h1 className="dashboard-title">MY DASHBOARD</h1>
        <button 
          className={`webcam-button ${isWebcamActive ? 'active' : ''}`}
          onClick={toggleWebcam}
        >
          <svg 
            className="camera-icon" 
            width="24" 
            height="24" 
            viewBox="0 0 24 24" 
            fill="none" 
            xmlns="http://www.w3.org/2000/svg"
          >
            <path 
              d="M23 19C23 19.5304 22.7893 20.0391 22.4142 20.4142C22.0391 20.7893 21.5304 21 21 21H3C2.46957 21 1.96086 20.7893 1.58579 20.4142C1.21071 20.0391 1 19.5304 1 19V8C1 7.46957 1.21071 6.96086 1.58579 6.58579C1.96086 6.21071 2.46957 6 3 6H7L9 3H15L17 6H21C21.5304 6 22.0391 6.21071 22.4142 6.58579C22.7893 6.96086 23 7.46957 23 8V19Z" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            />
            <path 
              d="M12 17C14.2091 17 16 15.2091 16 13C16 10.7909 14.2091 9 12 9C9.79086 9 8 10.7909 8 13C8 15.2091 9.79086 17 12 17Z" 
              stroke="currentColor" 
              strokeWidth="2" 
              strokeLinecap="round" 
              strokeLinejoin="round"
            />
          </svg>
          <span>{isWebcamActive ? 'Stop Webcam' : 'Start Webcam'}</span>
        </button>
        
        {isWebcamActive && (
          <div className="webcam-container">
            <img ref={videoRef} alt="Webcam feed" className="webcam-feed" />
            <div className="detections-list">
              {detections.map((detection, index) => (
                <div key={index} className="detection-item">{detection}</div>
              ))}
            </div>
          </div>
        )}
        
        <img src={backgroundImage} alt="" className="dashboard-background" />
      </main>
    </div>
  );
};

export default Dashboard; 