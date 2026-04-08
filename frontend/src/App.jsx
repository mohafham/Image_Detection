import React, { useState, useEffect } from 'react';
import axios from 'axios';

const RAW_BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8001';
const BACKEND_URL = RAW_BACKEND_URL.replace(/\/+$/, '');

function App() {
  const [backendStatus, setBackendStatus] = useState('checking');
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [dragOver, setDragOver] = useState(false);

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
    const interval = setInterval(checkBackendHealth, 5000); // Check every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const checkBackendHealth = async () => {
    try {
      await axios.get(`${BACKEND_URL}/health`, { timeout: 2000 });
      setBackendStatus('online');
    } catch (err) {
      setBackendStatus('offline');
    }
  };

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    processFile(file);
  };

  const handleDragOver = (event) => {
    event.preventDefault();
    setDragOver(true);
  };

  const handleDragLeave = (event) => {
    event.preventDefault();
    setDragOver(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    const file = event.dataTransfer.files[0];
    processFile(file);
  };

  const processFile = (file) => {
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select an image file (JPG, PNG)');
      return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('File size must be less than 10MB');
      return;
    }

    setSelectedFile(file);
    setError(null);
    setResult(null);

    // Create preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const analyzeImage = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const response = await axios.post(`${BACKEND_URL}/predict`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        timeout: 30000, // 30 seconds timeout
      });

      setResult(response.data);
    } catch (err) {
      if (err.code === 'ECONNABORTED') {
        setError('Request timeout. Please try again.');
      } else if (err.response) {
        setError(`Error: ${err.response.status} - ${err.response.data?.detail || 'Unknown error'}`);
      } else if (err.request) {
        setError('Cannot connect to backend. Make sure the server is running on port 8001.');
      } else {
        setError(`Error: ${err.message}`);
      }
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
  };

  const downloadHeatmap = () => {
    if (!result?.heatmap_base64) return;

    const link = document.createElement('a');
    link.href = `data:image/png;base64,${result.heatmap_base64}`;
    link.download = `gradcam_${selectedFile.name}`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <h1>🔍 AI Image Detector</h1>
        <p>Detect AI-generated images using deep learning</p>
      </header>

      {/* Status Bar */}
      <div className="status-bar">
        <div className="status-indicator">
          <div className={`status-dot ${backendStatus}`}></div>
          <span>
            Backend: {backendStatus === 'online' ? '✅ Connected' : '❌ Offline'}
          </span>
        </div>
        <div>
          <a href={`${BACKEND_URL}/docs`} target="_blank" rel="noopener noreferrer">
            📚 API Docs
          </a>
        </div>
      </div>

      {/* Main Container */}
      <div className="container">
        {/* Upload Section */}
        {!selectedFile && (
          <div
            className={`upload-section ${dragOver ? 'dragover' : ''}`}
            onClick={() => document.getElementById('fileInput').click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            <div className="upload-icon">📤</div>
            <h2>Upload an Image</h2>
            <p>Click to browse or drag & drop your image here</p>
            <p style={{ fontSize: '0.9rem', color: '#999' }}>
              Supported formats: JPG, JPEG, PNG (Max 10MB)
            </p>
            <input
              type="file"
              id="fileInput"
              className="upload-input"
              accept="image/jpeg,image/jpg,image/png"
              onChange={handleFileSelect}
            />
            <button className="upload-button">Choose File</button>
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="error">
            <strong>⚠️ Error:</strong> {error}
          </div>
        )}

        {/* Loading State */}
        {loading && (
          <div className="loading">
            <div className="spinner"></div>
            <h3>Analyzing image...</h3>
            <p>This may take a few seconds</p>
          </div>
        )}

        {/* Preview and Analyze Button */}
        {selectedFile && !loading && !result && (
          <div>
            <div className="result-card">
              <h3>📸 Selected Image</h3>
              <img src={previewUrl} alt="Preview" />
              <p>
                <strong>File:</strong> {selectedFile.name}<br />
                <strong>Size:</strong> {(selectedFile.size / 1024).toFixed(2)} KB
              </p>
            </div>
            <div className="actions">
              <button className="btn btn-primary" onClick={analyzeImage}>
                🔬 Analyze Image
              </button>
              <button className="btn btn-secondary" onClick={reset}>
                ❌ Cancel
              </button>
            </div>
          </div>
        )}

        {/* Results Display */}
        {result && (
          <div className="results">
            <div className="results-grid">
              {/* Original Image */}
              <div className="result-card">
                <h3>📸 Original Image</h3>
                <img src={previewUrl} alt="Original" />
              </div>

              {/* Grad-CAM Heatmap */}
              <div className="result-card">
                <h3>🔥 Grad-CAM Heatmap</h3>
                <img
                  src={`data:image/png;base64,${result.heatmap_base64}`}
                  alt="Heatmap"
                />
                <p style={{ fontSize: '0.9rem', color: '#666' }}>
                  Red areas show where the model focused
                </p>
              </div>

              {/* Prediction */}
              <div className="result-card">
                <h3>🎯 Prediction</h3>
                <div className={`verdict ${result.label === 'AI-GENERATED' ? 'ai-generated' : 'real'}`}>
                  {result.label === 'AI-GENERATED' ? '🤖 AI-Generated' : '✅ Real Image'}
                </div>
                <div className="confidence">
                  <div className="confidence-label">Confidence Level</div>
                  <div className="confidence-bar">
                    <div
                      className="confidence-fill"
                      style={{ width: `${result.confidence}%` }}
                    >
                      {result.confidence}%
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* Explanation */}
            <div className="explanation">
              <h3>💡 AI Explanation</h3>
              <p>{result.explanation}</p>
            </div>

            {/* Actions */}
            <div className="actions">
              <button className="btn btn-primary" onClick={downloadHeatmap}>
                📥 Download Heatmap
              </button>
              <button className="btn btn-primary" onClick={reset}>
                🔄 Analyze Another Image
              </button>
              <button
                className="btn btn-secondary"
                onClick={() => window.open(`${BACKEND_URL}/docs`, '_blank')}
              >
                📚 View API Docs
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <footer className="footer">
        <p>
          AI Image Detector v1.0.0 | Built with React & FastAPI |{' '}
          <a href={`${BACKEND_URL}/docs`} target="_blank" rel="noopener noreferrer">
            API Documentation
          </a>
        </p>
      </footer>
    </div>
  );
}

export default App;
