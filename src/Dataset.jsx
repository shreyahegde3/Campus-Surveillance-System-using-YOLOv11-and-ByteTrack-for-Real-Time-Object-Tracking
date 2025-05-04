import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import './Dataset.css';

const Dataset = () => {
  const navigate = useNavigate();

  useEffect(() => {
    // Add smooth scrolling effect
    const handleScroll = () => {
      const datasetCards = document.querySelectorAll('.dataset-card');
      
      datasetCards.forEach(card => {
        const cardPosition = card.getBoundingClientRect().top;
        const screenPosition = window.innerHeight / 1.3;
        
        if (cardPosition < screenPosition) {
          card.classList.add('show');
        }
      });
    };

    window.addEventListener('scroll', handleScroll);
    // Trigger once on load
    handleScroll();
    
    return () => {
      window.removeEventListener('scroll', handleScroll);
    };
  }, []);

  const handleBack = () => {
    navigate('/dashboard');
  };

  return (
    <div className="dataset-container">
      <button className="back-button" onClick={handleBack}>
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M15 18L9 12L15 6" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </button>
      <div className="dataset-header">
        <h1>Datasets</h1>
        <p>Explore the high-quality datasets that power our surveillance and object tracking system</p>
      </div>
      
      <div className="dataset-grid">
        <div className="dataset-card">
          <div className="dataset-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
          </div>
          <h2>COCO Dataset</h2>
          <p>Common Objects in Context (COCO) is a large-scale object detection, segmentation, and captioning dataset. COCO has several features:</p>
          <ul>
            <li>330K images (200K labeled)</li>
            <li>1.5 million object instances</li>
            <li>80 object categories</li>
            <li>5 captions per image</li>
          </ul>
          <div className="dataset-footer">
            <a href="https://cocodataset.org/" target="_blank" rel="noopener noreferrer" className="dataset-button">Learn More</a>
          </div>
        </div>
        
        <div className="dataset-card">
          <div className="dataset-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
              <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
              <line x1="12" y1="22.08" x2="12" y2="12"></line>
            </svg>
          </div>
          <h2>ABODA Dataset</h2>
          <p>Abandoned Objects Dataset (ABODA) is specifically designed for detecting abandoned objects in surveillance scenarios. Key features include:</p>
          <ul>
            <li>Diverse surveillance environments</li>
            <li>Multiple lighting conditions</li>
            <li>Various object types and sizes</li>
            <li>Annotated ground truth data</li>
          </ul>
          <div className="dataset-footer">
            <a href="#" className="dataset-button">Learn More</a>
          </div>
        </div>
        
        <div className="dataset-card">
          <div className="dataset-icon">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
              <polyline points="2 17 12 22 22 17"></polyline>
              <polyline points="2 12 12 17 22 12"></polyline>
            </svg>
          </div>
          <h2>Open Images V7</h2>
          <p>Open Images is a dataset of ~9M images annotated with image-level labels, object bounding boxes, object segmentation masks, and visual relationships. It contains:</p>
          <ul>
            <li>9 million+ images</li>
            <li>600 object classes</li>
            <li>15.8 million bounding boxes</li>
            <li>2.8 million instance segmentations</li>
          </ul>
          <div className="dataset-footer">
            <a href="https://storage.googleapis.com/openimages/web/index.html" target="_blank" rel="noopener noreferrer" className="dataset-button">Learn More</a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dataset;