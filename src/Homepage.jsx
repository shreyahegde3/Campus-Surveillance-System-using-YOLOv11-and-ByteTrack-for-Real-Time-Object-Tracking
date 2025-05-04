"use client"

import React, { useState } from "react"
import { useNavigate } from "react-router-dom"
import "./Homepage.css"
import cameraImage from "./camera.png"
import logoImage from "./camera_logo_black.jpg"

const Homepage = () => {
  const navigate = useNavigate()
  const [activeModal, setActiveModal] = useState(null)
  const [showModal, setShowModal] = useState(false)
  const [activeTab, setActiveTab] = useState("features")

  const openModal = (modalName) => {
    setActiveModal(modalName)
  }

  const closeModal = () => {
    setActiveModal(null)
  }

  const handleGetStarted = () => {
    navigate("/dashboard")
  }

  const handleSignIn = () => {
    navigate("/dashboard")
  }

  return (
    <div className="app-container">
      <nav className="navbar">
        <div className="nav-brand">
          <img src={logoImage} alt="SeekBot Logo" className="nav-logo-img" />
          <span className="nav-logo-text">SeekBot</span>
        </div>
        <div className="nav-right">
          <div className="nav-links">
            <a href="#" onClick={(e) => { e.preventDefault(); openModal('features'); }}>Features</a>
            <a href="#" onClick={(e) => { e.preventDefault(); openModal('pricing'); }}>Pricing</a>
            <a href="#" onClick={(e) => { e.preventDefault(); openModal('customers'); }}>Customers</a>
            <a href="#" onClick={(e) => { e.preventDefault(); openModal('contact'); }}>Contact Us</a>
          </div>
          <button className="sign-in-button" onClick={handleSignIn}>SIGN IN</button>
        </div>
      </nav>

      {/* Features Modal */}
      {activeModal === 'features' && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2 className="modal-title">Key Features</h2>
            <div className="modal-body">
              <div className="feature-item">
                <h3>Real-Time Monitoring</h3>
                <p>Advanced object detection and tracking for immediate threat identification.</p>
              </div>
              <div className="feature-item">
                <h3>Anomaly Detection</h3>
                <p>Detect unusual behavior patterns including sudden running and crowd formation.</p>
              </div>
              <div className="feature-item">
                <h3>Predictive Analytics</h3>
                <p>Historical pattern analysis for proactive threat prevention.</p>
              </div>
              <div className="feature-item">
                <h3>Instant Alerts</h3>
                <p>Real-time notification system for security personnel.</p>
              </div>
            </div>
            <button className="modal-close" aria-label="Close modal" onClick={closeModal}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Pricing Modal */}
      {activeModal === 'pricing' && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2 className="modal-title">Pricing Plans</h2>
            <div className="modal-body">
              <div className="pricing-item">
                <h3>Basic Plan</h3>
                <ul>
                  <li>Real-time monitoring</li>
                  <li>Basic alert system</li>
                  <li>24/7 Support</li>
                  <li>Up to 5 cameras</li>
                </ul>
                <button className="plan-button">Contact Sales</button>
              </div>
              <div className="pricing-item featured">
                <h3>Professional Plan</h3>
                <ul>
                  <li>All Basic features</li>
                  <li>Anomaly detection</li>
                  <li>Pattern analysis</li>
                  <li>Up to 20 cameras</li>
                  <li>Priority support</li>
                </ul>
                <button className="plan-button">Contact Sales</button>
              </div>
              <div className="pricing-item">
                <h3>Enterprise Plan</h3>
                <ul>
                  <li>All Professional features</li>
                  <li>Custom integration</li>
                  <li>Unlimited cameras</li>
                  <li>Dedicated support team</li>
                </ul>
                <button className="plan-button">Contact Sales</button>
              </div>
            </div>
            <button className="modal-close" aria-label="Close modal" onClick={closeModal}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Customers Modal */}
      {activeModal === 'customers' && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2 className="modal-title">Our Customers</h2>
            <div className="modal-body">
              <div className="customer-item">
                <span className="customer-icon">🏛️</span>
                <h3>Universities</h3>
                <p>Protecting students and faculty across multiple campuses with state-of-the-art surveillance.</p>
              </div>
              <div className="customer-item">
                <span className="customer-icon">🎓</span>
                <h3>Colleges</h3>
                <p>Ensuring safety in educational environments with intelligent monitoring systems.</p>
              </div>
              <div className="customer-item">
                <span className="customer-icon">🔬</span>
                <h3>Research Institutes</h3>
                <p>Securing valuable research facilities with advanced tracking capabilities.</p>
              </div>
            </div>
            <button className="modal-close" aria-label="Close modal" onClick={closeModal}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Contact Modal */}
      {activeModal === 'contact' && (
        <div className="modal-overlay" onClick={closeModal}>
          <div className="modal-content" onClick={e => e.stopPropagation()}>
            <h2 className="modal-title">Contact Our Team</h2>
            <div className="modal-body">
              <div className="contact-item">
                <h3>Pranav Rao</h3>
                <p>
                  <svg className="contact-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72 12.84 12.84 0 00.7 2.81 2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45 12.84 12.84 0 002.81.7A2 2 0 0122 16.92z"></path>
                  </svg>
                  +91 82177 09653
                </p>
                <a href="mailto:prao52623@gmail.com">
                  <svg className="contact-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                    <polyline points="22,6 12,13 2,6"></polyline>
                  </svg>
                  prao52623@gmail.com
                </a>
              </div>
              <div className="contact-item">
                <h3>Pranav Acharya</h3>
                <p>
                  <svg className="contact-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72 12.84 12.84 0 00.7 2.81 2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45 12.84 12.84 0 002.81.7A2 2 0 0122 16.92z"></path>
                  </svg>
                  +91 7022939074
                </p>
                <a href="mailto:pranavacharya360@gmail.com">
                  <svg className="contact-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                    <polyline points="22,6 12,13 2,6"></polyline>
                  </svg>
                  pranavacharya360@gmail.com
                </a>
              </div>
              <div className="contact-item">
                <h3>Rishika N</h3>
                <p>
                  <svg className="contact-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72 12.84 12.84 0 00.7 2.81 2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45 12.84 12.84 0 002.81.7A2 2 0 0122 16.92z"></path>
                  </svg>
                  +91 70198 25753
                </p>
                <a href="mailto:rishikanaarayan2003@gmail.com">
                  <svg className="contact-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                    <polyline points="22,6 12,13 2,6"></polyline>
                  </svg>
                  rishikanaarayan2003@gmail.com
                </a>
              </div>
              <div className="contact-item">
                <h3>Shreya Hegde</h3>
                <p>
                  <svg className="contact-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 16.92v3a2 2 0 01-2.18 2 19.79 19.79 0 01-8.63-3.07 19.5 19.5 0 01-6-6 19.79 19.79 0 01-3.07-8.67A2 2 0 014.11 2h3a2 2 0 012 1.72 12.84 12.84 0 00.7 2.81 2 2 0 01-.45 2.11L8.09 9.91a16 16 0 006 6l1.27-1.27a2 2 0 012.11-.45 12.84 12.84 0 002.81.7A2 2 0 0122 16.92z"></path>
                  </svg>
                  +91 76187 54280
                </p>
                <a href="mailto:shreya.m.hegde@gmail.com">
                  <svg className="contact-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
                    <polyline points="22,6 12,13 2,6"></polyline>
                  </svg>
                  shreya.m.hegde@gmail.com
                </a>
              </div>
            </div>
            <button className="modal-close" aria-label="Close modal" onClick={closeModal}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M18 6L6 18M6 6l12 12"></path>
              </svg>
            </button>
          </div>
        </div>
      )}

      <div className="homepage">
        <div className="content-container">
          <div className="text-content">
            <h1 className="main-heading">
              THE CAMPUS
              <br />
              SURVEILLANCE
            </h1>

            <p className="description">
              Advanced object detection and tracking capabilities for enhanced campus security. Monitor and protect your
              campus with our cutting-edge technology.
            </p>

            <div className="button-container">
              <button type="button" className="submit-button" onClick={handleGetStarted}>
                GET STARTED
              </button>
            </div>
          </div>

          <div className="image-container">
            <img src={cameraImage} alt="Campus Surveillance Camera" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default Homepage
