# React Frontend for AI Image Detector

A modern, responsive React frontend built with Vite for the AI Image Detection system.

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)

Run both backend and frontend:
```bash
.\start_react.ps1
```

### Option 2: Manual Startup

**Terminal 1 - Backend:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## 📦 Installation

Install Node.js dependencies:
```bash
cd frontend
npm install
```

This will install:
- React 18.2.0
- Axios (for API calls)
- Vite (dev server & build tool)

## ✨ Features

- ✅ **Modern React UI** - Built with React 18 hooks
- ✅ **Drag & Drop Upload** - Easy file selection
- ✅ **Real-time Backend Status** - Live connection monitoring
- ✅ **Progress Indicators** - Visual feedback during analysis
- ✅ **Responsive Design** - Works on all screen sizes
- ✅ **Error Handling** - Comprehensive error messages
- ✅ **Download Heatmaps** - Save Grad-CAM visualizations
- ✅ **Beautiful Gradient UI** - Modern, professional design

## 🎨 UI Components

### Header
- Gradient background
- App title and description

### Status Bar
- Live backend connection indicator
- Quick link to API docs

### Upload Section
- Click to browse files
- Drag & drop support
- File type validation (JPG, PNG)
- 10MB size limit

### Results Display
- Three-column layout:
  1. Original image
  2. Grad-CAM heatmap
  3. Prediction & confidence

### Explanation Section
- AI-generated explanation
- Color-coded border (green for Real, red for Fake)

### Action Buttons
- Download heatmap
- Analyze another image
- View API documentation

## 🛠️ Technology Stack

- **React 18** - UI framework
- **Vite** - Build tool & dev server
- **Axios** - HTTP client
- **CSS3** - Styling with gradients & animations

## 🎯 API Integration

The frontend connects to the FastAPI backend:

```javascript
const BACKEND_URL = 'http://localhost:8001';

// Health check
GET /health

// Image prediction
POST /predict
```

## 📱 Responsive Design

- **Desktop**: Full three-column layout
- **Tablet**: Optimized spacing
- **Mobile**: Single-column stack

## 🔧 Configuration

### Change Backend URL

Edit `src/App.jsx`:
```javascript
const BACKEND_URL = 'http://localhost:8001'; // Change port if needed
```

### Change Frontend Port

Edit `vite.config.js`:
```javascript
server: {
  port: 8501, // Change to your preferred port
}
```

## 📂 File Structure

```
frontend/
├── index.html              # HTML template
├── package.json            # Dependencies
├── vite.config.js          # Vite configuration
├── src/
│   ├── main.jsx           # React entry point
│   ├── App.jsx            # Main App component
│   └── index.css          # Global styles
└── app.py                 # Old Streamlit app (not used)
```

## 🚀 Available Scripts

```bash
# Development server
npm run dev

# Production build
npm run build

# Preview production build
npm run preview
```

## 🎨 Features Breakdown

### 1. File Upload
- Visual upload zone with hover effects
- Drag & drop functionality
- File validation (type & size)
- Image preview

### 2. Analysis
- Loading spinner with status text
- API request with 30s timeout
- Progress feedback
- Error recovery

### 3. Results
- Original image display
- Grad-CAM heatmap visualization
- Verdict with color coding
- Confidence bar with percentage
- AI explanation

### 4. Error Handling
- Connection errors
- Timeout errors
- Invalid file types
- File size limits
- Backend offline detection

## 🎯 User Flow

1. **Upload Image** → Click or drag & drop
2. **Preview** → See file details
3. **Analyze** → Click analyze button
4. **Wait** → See progress indicator
5. **View Results** → See prediction, heatmap, explanation
6. **Download** → Save heatmap (optional)
7. **Repeat** → Analyze another image

## 🔍 Troubleshooting

### Backend Not Connecting

```bash
# Check backend is running
curl http://localhost:8001/health

# Start backend
cd backend
uvicorn main:app --port 8001
```

### NPM Install Fails

```bash
# Clear cache
npm cache clean --force

# Delete node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### Port Already in Use

Change port in `vite.config.js` or kill process:
```bash
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F
```

## 📊 Performance

- **Initial Load**: <1s
- **File Upload**: Instant
- **API Request**: 1-3s
- **Total Analysis Time**: 2-5s

## 🎉 Advantages Over Streamlit

- ✅ Faster load times
- ✅ More customizable UI
- ✅ Better mobile support
- ✅ Standard web technologies
- ✅ Easier deployment
- ✅ More professional appearance

## 📝 Notes

- Frontend runs on port **8501** (same as Streamlit)
- Backend must run on port **8001**
- Supports modern browsers (Chrome, Firefox, Safari, Edge)
- Node.js 16+ required

## 🚀 Ready to Use!

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:8501 in your browser!
