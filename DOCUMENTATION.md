# 🌱 AgroSentinel - Technical Documentation

> **AI-Powered Crop Disease Detection System for Indian Farmers**
> 
> Version: 1.0.0 | Last Updated: November 2025

---

## 📑 Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Machine Learning Model](#4-machine-learning-model)
5. [Backend API](#5-backend-api)
6. [Frontend Application](#6-frontend-application)
7. [Database Schema](#7-database-schema)
8. [External Integrations](#8-external-integrations)
9. [Installation & Setup](#9-installation--setup)
10. [API Reference](#10-api-reference)
11. [Offline Capabilities](#11-offline-capabilities)
12. [Multi-Language Support](#12-multi-language-support)
13. [Security Considerations](#13-security-considerations)
14. [Performance Metrics](#14-performance-metrics)
15. [Future Enhancements](#15-future-enhancements)

---

## 1. Project Overview

### 1.1 Problem Statement

Indian farmers face significant crop losses (estimated ₹50,000+ crores annually) due to plant diseases. Traditional disease identification methods require:
- Expert agricultural knowledge
- Time-consuming manual inspection
- Delayed treatment leading to crop damage

### 1.2 Solution

**AgroSentinel** is an AI-powered mobile-first web application that enables farmers to:
- **Instantly identify** crop diseases by capturing leaf images
- **Receive treatment recommendations** based on ICAR (Indian Council of Agricultural Research) guidelines
- **Assess risk levels** using real-time weather correlation
- **Get expert guidance** through an AI-powered chat assistant
- **Monitor field health** via drone view visualization

### 1.3 Key Features

| Feature | Description |
|---------|-------------|
| 🔬 AI Disease Detection | EfficientNet-B3 model with 99.84% accuracy |
| 🌤️ Weather-Correlated Risk | Real-time weather integration for risk assessment |
| 💊 ICAR Treatment Database | Authentic medication recommendations |
| 🗺️ Drone View | Satellite imagery with disease clustering |
| 🤖 AI Chat Assistant | Gemma 2 powered farming assistant |

### 1.4 Supported Crops & Diseases

| Crop | Diseases Detected | Count |
|------|-------------------|-------|
| 🍅 Tomato | Late Blight, Early Blight, Bacterial Spot, Leaf Mold, Mosaic Virus, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Healthy | 10 |
| 🥔 Potato | Late Blight, Early Blight, Healthy | 3 |
| 🌶️ Pepper | Bacterial Spot, Healthy | 2 |
| **Total** | | **15 classes** |

---

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        CLIENT LAYER                             │
│  React 18 PWA + Tailwind CSS + Service Worker + IndexedDB       │
└─────────────────────────────────┬───────────────────────────────┘
                                  │ REST API (HTTPS)
┌─────────────────────────────────▼───────────────────────────────┐
│                        API LAYER                                │
│  FastAPI + CORS Middleware + Request Validation                 │
└─────────────────────────────────┬───────────────────────────────┘
                                  │
        ┌─────────────────────────┼─────────────────────────┐
        ▼                         ▼                         ▼
┌───────────────┐       ┌─────────────────┐       ┌─────────────────┐
│  ML Inference │       │  Gemma Chat     │       │ Weather Service │
│  ONNX Runtime │       │  Google AI      │       │ OpenWeatherMap  │
└───────────────┘       └─────────────────┘       └─────────────────┘
        │                         │                         │
        └─────────────────────────┼─────────────────────────┘
                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  MongoDB (Motor Async) + ICAR Treatment Database                │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Component Interaction Flow

```
User Input → Frontend → API Gateway → Service Layer → Data Layer → Response
     ↑                                      │
     └──────────────────────────────────────┘
```

### 2.3 Directory Structure

```
AgroSentinel/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI application entry
│   │   ├── config.py            # Environment configuration
│   │   ├── models/
│   │   │   └── schemas.py       # Pydantic data models
│   │   ├── routes/
│   │   │   ├── diagnosis.py     # Disease detection endpoints
│   │   │   └── chat.py          # Chat assistant endpoints
│   │   └── services/
│   │       ├── inference.py     # ML inference engine
│   │       ├── weather_service.py
│   │       ├── risk_engine.py   # Risk calculation
│   │       ├── disease_data.py  # Treatment database
│   │       ├── gemma_chat.py    # Gemma AI integration
│   │       ├── database.py      # MongoDB operations
│   │       └── translations.py  # Multi-language support
│   ├── models/
│   │   ├── crop_disease_model.onnx  # Trained model (43.8 MB)
│   │   ├── class_names.json
│   │   └── inference_config.json
│   ├── requirements.txt
│   └── Procfile
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main application component
│   │   ├── main.jsx             # Entry point
│   │   ├── index.css            # Global styles
│   │   ├── pages/
│   │   │   ├── Home.jsx
│   │   │   ├── Scanner.jsx      # Disease scanner
│   │   │   ├── ChatAssistant.jsx
│   │   │   ├── DroneView.jsx    # Map visualization
│   │   │   └── History.jsx
│   │   ├── components/
│   │   │   ├── Navbar.jsx
│   │   │   ├── Header.jsx
│   │   │   ├── ResultCard.jsx
│   │   │   └── OfflineIndicator.jsx
│   │   ├── services/
│   │   │   ├── api.js           # API client
│   │   │   └── offlineStorage.js # IndexedDB operations
│   │   ├── store/
│   │   │   └── useStore.js      # Zustand state management
│   │   └── i18n/
│   │       ├── index.js
│   │       ├── LanguageContext.jsx
│   │       └── translations.js  # 5 language translations
│   ├── public/
│   │   ├── sw.js                # Service worker
│   │   └── manifest.json        # PWA manifest
│   ├── package.json
│   └── vite.config.js
└── training/
    ├── train_model.py           # Model training script
    ├── train_model_v2.py        # Improved training
    ├── export_onnx.py           # ONNX export
    ├── evaluate_model.py        # Model evaluation
    └── datasets/
        └── agrosentinel/        # Training data
            ├── train/           # 16,504 images
            ├── val/             # 3,089 images
            └── test/            # 1,045 images
```

---

## 3. Technology Stack

### 3.1 Frontend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| React | 18.x | UI framework |
| Vite | 5.x | Build tool & dev server |
| Tailwind CSS | 3.x | Utility-first CSS framework |
| React Router | 6.x | Client-side routing |
| Zustand | 4.x | State management |
| Framer Motion | 10.x | Animations |
| Leaflet | 1.9.x | Interactive maps |
| react-webcam | 7.x | Camera access |
| Axios | 1.x | HTTP client |

### 3.2 Backend Technologies

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.11+ | Runtime environment |
| FastAPI | 0.100+ | Web framework |
| ONNX Runtime | 1.16+ | ML inference |
| Motor | 3.x | Async MongoDB driver |
| Pydantic | 2.x | Data validation |
| aiohttp | 3.x | Async HTTP client |
| google-generativeai | 0.3+ | Gemma AI SDK |
| Pillow | 10.x | Image processing |
| NumPy | 1.24+ | Numerical computing |

### 3.3 ML/AI Technologies

| Technology | Purpose |
|------------|---------|
| PyTorch | Model training |
| EfficientNet-B3 | Base architecture |
| ONNX | Model format for inference |
| Google Gemma 2 | Chat AI |

### 3.4 Infrastructure

| Service | Purpose |
|---------|---------|
| MongoDB Atlas | Cloud database |
| OpenWeatherMap API | Weather data |
| Railway/Render | Backend hosting |
| Vercel/Netlify | Frontend hosting (optional) |

---

## 4. Machine Learning Model

### 4.1 Model Architecture

```
EfficientNet-B3 (Transfer Learning)
├── Input: 260×260×3 RGB Image
├── Backbone: EfficientNet-B3 (ImageNet pretrained)
├── Custom Classifier:
│   ├── Dropout (0.4)
│   ├── Linear (1536 → 512)
│   ├── ReLU
│   ├── Dropout (0.3)
│   └── Linear (512 → 15)
└── Output: 15 class probabilities
```

### 4.2 Training Configuration

| Parameter | Value |
|-----------|-------|
| Base Model | EfficientNet-B3 |
| Input Size | 260×260 pixels |
| Batch Size | 24 |
| Learning Rate | 0.0005 |
| Optimizer | AdamW |
| Loss Function | Focal Loss (γ=2.0) |
| Label Smoothing | 0.1 |
| Mixup Alpha | 0.2 |
| Epochs | 30 (Early stopping) |
| Best Epoch | 9 |

### 4.3 Data Augmentation

**Training Augmentations:**
- Random Resized Crop (scale: 0.7-1.0)
- Random Horizontal Flip (p=0.5)
- Random Vertical Flip (p=0.3)
- Random Rotation (±45°)
- Color Jitter (brightness, contrast, saturation, hue)
- Random Affine Transform
- Random Perspective
- Gaussian Blur
- Random Erasing (p=0.25)

**Test-Time Augmentation (TTA):**
1. Original image
2. Horizontal flip
3. Slight rotation (-10°)
4. Brightness adjustment (+10%)
5. Center crop zoom (90%)

### 4.4 Dataset Statistics

| Split | Images | Purpose |
|-------|--------|---------|
| Training | 16,504 | Model training |
| Validation | 3,089 | Hyperparameter tuning |
| Test | 1,045 | Final evaluation |
| **Total** | **20,638** | |

**Source:** PlantVillage Dataset (curated for Indian crops)

### 4.5 Model Performance

| Metric | Value |
|--------|-------|
| Validation Accuracy | 99.84% |
| Test Accuracy | 99.5%+ |
| Model Size (ONNX) | 43.8 MB |
| Parameters | 11.49 M |
| Inference Time (CPU) | ~19 ms |
| Inference Time (TTA) | ~87 ms |
| FPS (Single) | ~52 FPS |
| FPS (TTA) | ~11 FPS |

### 4.6 Class-wise Performance

All 15 classes achieve >99% precision and recall with minimal confusion between similar diseases.

---

## 5. Backend API

### 5.1 Application Configuration

```python
# backend/app/config.py
class Settings:
    mongodb_uri: str          # MongoDB connection string
    openweathermap_api_key: str  # Weather API key
    ai_api_key: str           # Google AI API key
    model_path: str           # Path to ONNX model
```

### 5.2 API Endpoints Overview

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/predict` | Quick disease prediction |
| POST | `/api/analyze` | Full analysis with weather & risk |
| GET | `/api/weather` | Get weather data |
| GET | `/api/remedies/{disease}` | Get treatment info |
| GET | `/api/history/location` | Location-based scan history |
| POST | `/api/chat/send` | Send chat message |
| GET | `/api/chat/suggestions` | Get quick questions |
| GET | `/api/languages` | Get supported languages |
| GET | `/health` | Health check |

### 5.3 Core Services

#### 5.3.1 Inference Service (`inference.py`)

```python
class EfficientNetInference:
    def __init__(self, model_path: str):
        # Load ONNX model with GPU/CPU fallback
        
    def smart_preprocess(self, image: Image) -> Image:
        # Handle various aspect ratios
        # Center crop for borders/watermarks
        
    def get_tta_images(self, image: Image) -> list:
        # Generate 5 TTA variants
        
    def predict(self, image_bytes: bytes) -> tuple:
        # Returns: (disease, confidence, top5, is_confident)
```

#### 5.3.2 Risk Engine (`risk_engine.py`)

```python
class RiskEngine:
    @staticmethod
    def calculate_risk(disease, confidence, weather) -> tuple:
        # Risk = (confidence × 0.6) + (environmental_risk × 0.4)
        # Returns: (risk_score, risk_level)
        # Levels: LOW, MODERATE, HIGH, CRITICAL, HEALTHY
```

**Disease-Specific Thresholds:**
| Disease | Humidity Threshold | Temp Range |
|---------|-------------------|------------|
| Late Blight | 85% | 10-25°C |
| Early Blight | 60% | 20-30°C |
| Bacterial Spot | 70% | 20-35°C |
| Spider Mites | 40% | 25-40°C |

#### 5.3.3 Weather Service (`weather_service.py`)

```python
class WeatherService:
    async def get_weather(self, lat, lon) -> WeatherData:
        # Fetches from OpenWeatherMap API
        # Returns: humidity, temperature, description, wind_speed
        # Fallback data for offline/demo mode
```

#### 5.3.4 Gemma Chat Service (`gemma_chat.py`)

```python
class GemmaChatService:
    def __init__(self, api_key: str):
        # Initialize Gemma 2 model
        # Load agricultural system context
        
    async def send_message(self, message, language, session_id) -> dict:
        # Process user query
        # Returns: response, suggestions, intent, detected_disease
```

**System Context Includes:**
- All 15 supported diseases with symptoms
- Treatment protocols (chemical + organic)
- Crop cultivation guides for India
- Weather-based advice
- Response guidelines for farmers

---

## 6. Frontend Application

### 6.1 Application Structure

```jsx
<LanguageProvider>
  <BrowserRouter>
    <OfflineIndicator />
    <Header />
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/scan" element={<Scanner />} />
      <Route path="/history" element={<History />} />
      <Route path="/drone" element={<DroneView />} />
      <Route path="/chat" element={<ChatAssistant />} />
    </Routes>
    <Navbar />
  </BrowserRouter>
</LanguageProvider>
```

### 6.2 Key Components

#### 6.2.1 Scanner Page

**Features:**
- Camera capture with rear-facing preference
- Image upload from gallery
- Real-time GPS location tracking
- Offline scan queuing
- Result display with treatment recommendations

**Flow:**
1. User captures/uploads image
2. Get GPS coordinates
3. Check online status
4. If online: Send to API → Display results
5. If offline: Save to IndexedDB → Show pending message

#### 6.2.2 Chat Assistant

**Features:**
- Real-time AI conversation
- Multi-language support
- Quick question suggestions
- Markdown formatting for responses
- Session-based conversation history

#### 6.2.3 Drone View

**Features:**
- Interactive Leaflet map
- Disease markers with risk colors
- Simulated drone scanning animation
- Clustered infection visualization
- Location-based history overlay

### 6.3 State Management (Zustand)

```javascript
const useStore = create((set) => ({
  // Theme
  theme: 'system', // 'light' | 'dark' | 'system'
  
  // Scanner state
  result: null,
  loading: false,
  location: null,
  
  // Actions
  setResult: (result) => set({ result }),
  setLoading: (loading) => set({ loading }),
  setLocation: (location) => set({ location }),
  clearResult: () => set({ result: null }),
}))
```

### 6.4 API Client (`api.js`)

```javascript
// Offline-aware API service
const api = axios.create({
  baseURL: '/api',
  timeout: 30000
})

// Online status tracking
window.addEventListener('online', syncPendingScans)
window.addEventListener('offline', () => {})

// Main functions
export const analyzeWithRisk = async (file, lat, lon) => {
  if (isOnline) {
    // Try API request
  } else {
    // Save to IndexedDB for later
  }
}
```

---

## 7. Database Schema

### 7.1 MongoDB Collections

#### 7.1.1 Diagnoses Collection

```javascript
{
  _id: ObjectId,
  user_id: String (optional),
  location: {
    latitude: Number,
    longitude: Number
  },
  disease: String,           // e.g., "tomato_late_blight"
  confidence: Number,        // 0.0 - 1.0
  weather: {
    humidity: Number,
    temperature: Number,
    description: String,
    wind_speed: Number
  },
  risk_score: Number,        // 0.0 - 1.0
  remedy: {
    spray: String,
    repeat: String,
    precautions: String,
    organic: String
  },
  image_url: String (optional),
  created_at: ISODate
}
```

#### 7.1.2 Chat History Collection

```javascript
{
  _id: ObjectId,
  session_id: String,
  user_message: String,
  assistant_response: String,
  language: String,          // "en", "hi", "te", "ta", "kn"
  intent: String,
  model: String,             // "gemma" or "basic"
  timestamp: ISODate
}
```

### 7.2 IndexedDB Stores (Client-Side)

| Store | Purpose | Key |
|-------|---------|-----|
| `scans` | Completed scan results | Auto-increment ID |
| `pending_uploads` | Offline scans waiting to sync | Auto-increment ID |
| `api_cache` | Cached API responses | URL |
| `settings` | User preferences | Key name |

---

## 8. External Integrations

### 8.1 OpenWeatherMap API

**Endpoint:** `https://api.openweathermap.org/data/2.5/weather`

**Parameters:**
- `lat`: Latitude
- `lon`: Longitude
- `appid`: API key
- `units`: metric

**Response Used:**
```json
{
  "main": {
    "humidity": 65,
    "temp": 28.5
  },
  "weather": [{ "description": "partly cloudy" }],
  "wind": { "speed": 3.5 }
}
```

**Model:** `gemma-2-9b-it`

**Configuration:**
```python
generation_config = {
    'temperature': 0.7,
    'top_p': 0.8,
    'top_k': 40,
    'max_output_tokens': 1024
}
```

**System Prompt:** Comprehensive agricultural knowledge base including:
- All supported diseases with symptoms and treatments
- Crop cultivation guides for Indian conditions
- Weather-based advice
- Both chemical and organic treatment options

---

## 9. Installation & Setup

### 9.1 Prerequisites

- Python 3.11+
- Node.js 18+
- MongoDB (local or Atlas)
- Git

### 9.2 Backend Setup

```bash
# Clone repository
git clone https://github.com/Kranthi-Swaroop/AgroVision.git
cd AgroVision

# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Create .env file
echo "MONGODB_URI=mongodb://localhost:27017/agrosentinel" > .env
echo "OPENWEATHERMAP_API_KEY=your_key_here" >> .env
echo "AI_API_KEY=your_key_here" >> .env

# Run backend
uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
```

### 9.3 Frontend Setup

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Create .env file (optional, defaults to /api)
echo "VITE_API_URL=http://localhost:8001/api" > .env

# Run development server
npm run dev

# Build for production
npm run build
```

### 9.4 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `MONGODB_URI` | Yes | MongoDB connection string |
| `OPENWEATHERMAP_API_KEY` | Yes | Weather API key (free tier available) |
| `AI_API_KEY` | Yes | Google AI API key |
| `VITE_API_URL` | No | Backend API URL (default: `/api`) |

### 9.5 Deployment

**Railway/Render (Backend):**
```yaml
# railway.json / render.yaml
services:
  - type: web
    name: agrosentinel-api
    env: python
    buildCommand: pip install -r requirements.txt
    startCommand: uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

**Static Frontend:** Build with `npm run build`, deploy `dist/` folder to any static host.

---

## 10. API Reference

### 10.1 Disease Prediction

#### Quick Predict

```http
POST /api/predict
Content-Type: multipart/form-data
```

**Parameters:**
| Name | Type | Required | Description |
|------|------|----------|-------------|
| file | File | Yes | Image file (JPEG/PNG) |
| lang | Query | No | Language code (default: "en") |

**Response:**
```json
{
  "disease": "tomato_late_blight",
  "display_name": "Tomato Late Blight",
  "confidence": 0.9847,
  "crop": "tomato",
  "is_healthy": false,
  "severity": "critical",
  "top_predictions": [
    {"class": "tomato_late_blight", "confidence": 0.9847},
    {"class": "tomato_early_blight", "confidence": 0.0102}
  ],
  "is_confident": true
}
```

#### Full Analysis

```http
POST /api/analyze?latitude=17.385&longitude=78.486&lang=en
Content-Type: multipart/form-data
```

**Response:**
```json
{
  "disease": "tomato_late_blight",
  "display_name": "Tomato Late Blight",
  "confidence": 0.9847,
  "crop": "tomato",
  "is_healthy": false,
  "severity": "critical",
  "weather": {
    "humidity": 85,
    "temperature": 22.5,
    "description": "light rain",
    "wind_speed": 3.2
  },
  "risk_score": 0.892,
  "risk_level": "CRITICAL",
  "risk_level_display": "Critical Risk",
  "treatment": {
    "spray": "Metalaxyl + Mancozeb @ 2.5g/L",
    "repeat": "5-7 days",
    "precautions": "Destroy infected plants immediately",
    "organic": "Copper fungicide, remove infected parts"
  },
  "remedy": {...},
  "is_confident": true,
  "warning": null
}
```

### 10.2 Chat API

```http
POST /api/chat/send
Content-Type: application/json
```

**Request Body:**
```json
{
  "message": "How to treat late blight in tomatoes?",
  "language": "en",
  "session_id": "session_12345"
}
```

**Response:**
```json
{
  "response": "Late blight is a serious fungal disease...",
  "suggestions": [
    "What about organic treatment?",
    "How to prevent late blight?"
  ],
  "intent": "treatment_query",
  "detected_disease": "late_blight",
  "detected_crop": "tomato",
  "language": "en",
  "timestamp": "2025-11-26T10:30:00Z"
}
```

### 10.3 Weather API

```http
GET /api/weather?latitude=17.385&longitude=78.486
```

**Response:**
```json
{
  "humidity": 72,
  "temperature": 28.5,
  "description": "partly cloudy",
  "wind_speed": 4.2
}
```

### 10.4 Treatment Info

```http
GET /api/remedies/tomato_late_blight
```

**Response:**
```json
{
  "disease": "tomato_late_blight",
  "remedy": {
    "spray": "Metalaxyl + Mancozeb @ 2.5g/L or Cymoxanil + Mancozeb @ 3g/L",
    "repeat": "5-7 days during outbreak",
    "precautions": "URGENT: Destroy infected plants immediately",
    "organic": "Copper fungicide, Bordeaux mixture",
    "severity": "critical",
    "yield_loss": "70-100% if untreated"
  }
}
```

---

## 11. Offline Capabilities

### 11.1 Service Worker

**Cached Resources:**
- Static assets (HTML, CSS, JS)
- Navigation routes (SPA support)
- API responses (weather, languages)

**Strategies:**
- Static assets: Cache-first
- API calls: Network-first with cache fallback
- POST requests: Queue for background sync

### 11.2 IndexedDB Operations

```javascript
// Save scan for later upload
await offlineStorage.savePendingScan(file, { latitude, longitude })

// Get pending scans
const pending = await offlineStorage.getPendingScans()

// Sync when online
window.addEventListener('online', async () => {
  const pending = await offlineStorage.getPendingScans()
  for (const scan of pending) {
    await uploadScan(scan)
    await offlineStorage.removePendingScan(scan.id)
  }
})
```

### 11.3 Offline User Experience

1. **Offline Indicator:** Visual banner when disconnected
2. **Pending Queue:** Scans saved locally with timestamp
3. **Auto-Sync:** Automatic upload when connection restored
4. **Cached Results:** Previously fetched data available offline

---

## 12. Multi-Language Support

### 12.1 Supported Languages

| Code | Language | Native Name |
|------|----------|-------------|
| en | English | English |
| hi | Hindi | हिंदी |
| te | Telugu | తెలుగు |
| ta | Tamil | தமிழ் |
| kn | Kannada | ಕನ್ನಡ |

### 12.2 Implementation

**Frontend (React Context):**
```jsx
const LanguageContext = createContext()

export const LanguageProvider = ({ children }) => {
  const [language, setLanguage] = useState('en')
  const t = (key) => translations[language][key] || key
  
  return (
    <LanguageContext.Provider value={{ language, setLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  )
}
```

**Backend (Translation Service):**
```python
def get_disease_name(disease: str, lang: str) -> str:
    return DISEASE_TRANSLATIONS.get(lang, {}).get(disease, disease)

def get_risk_level_name(level: str, lang: str) -> str:
    return RISK_TRANSLATIONS.get(lang, {}).get(level, level)
```

### 12.3 Translation Coverage

- UI labels and navigation
- Disease names
- Treatment descriptions
- Risk level indicators
- Chat responses (via Gemini)
- Error messages

---

## 13. Security Considerations

### 13.1 API Security

- **CORS:** Configured for allowed origins
- **Input Validation:** Pydantic models for request validation
- **File Validation:** Content-type and size checks
- **Rate Limiting:** Recommended for production

### 13.2 Data Privacy

- **No PII Collection:** Only location coordinates stored
- **Session-Based Chat:** No persistent user identification
- **Local Storage:** Sensitive data stored client-side in IndexedDB

### 13.3 API Key Protection

- Environment variables for all API keys
- Keys never exposed to frontend
- Backend proxies all external API calls

### 13.4 Recommendations for Production

1. Enable HTTPS only
2. Implement rate limiting
3. Add authentication for history features
4. Set up API key rotation
5. Enable MongoDB authentication
6. Add request logging and monitoring

---

## 14. Performance Metrics

### 14.1 Model Performance

| Metric | Value |
|--------|-------|
| Accuracy | 99.84% |
| Inference (Single) | ~19 ms |
| Inference (TTA) | ~87 ms |
| FPS (Real-time) | ~52 FPS |
| Model Size | 43.8 MB |

### 14.2 API Performance

| Endpoint | Avg Response Time |
|----------|-------------------|
| /api/predict | ~150 ms |
| /api/analyze | ~300 ms |
| /api/chat | ~500-1000 ms |
| /api/weather | ~200 ms |

### 14.3 Frontend Performance

| Metric | Value |
|--------|-------|
| First Contentful Paint | < 1.5s |
| Time to Interactive | < 3s |
| Lighthouse Score | 90+ |
| Bundle Size (gzipped) | ~200 KB |

---

## 15. Future Enhancements

### 15.1 Short-term (v1.1)

- [ ] User authentication system
- [ ] Scan history with images
- [ ] Push notifications for risk alerts
- [ ] More language support (Marathi, Bengali)

### 15.2 Medium-term (v2.0)

- [ ] Real drone integration (DJI SDK)
- [ ] Crop yield prediction model
- [ ] Soil health analysis
- [ ] Marketplace for agricultural products
- [ ] Expert consultation booking

### 15.3 Long-term (v3.0)

- [ ] IoT sensor integration
- [ ] Automated irrigation control
- [ ] Satellite imagery analysis
- [ ] Community disease alerts
- [ ] Government scheme integration

---

## 📞 Support & Contact

**Team:** Overclocked Minds

**Repository:** [GitHub - AgroSentinel](https://github.com/Kranthi-Swaroop/AgroVision)

**Issues:** Report bugs via GitHub Issues

---

## 📄 License

MIT License - Free for educational and commercial use.

---

*Documentation last updated: November 2025*
