# 🎨 Brand AI Generator - Adobe Express Add-on

> **AI-Powered Marketing Poster Generator** - Transform your brand assets into stunning marketing posters with intelligent AI-powered design generation and custom text editing.

## 🌟 Overview

**Brand AI Generator** is a powerful Adobe Express add-on that leverages cutting-edge AI technology to automatically create professional marketing posters from your brand assets. Using the advanced **BrandDiffusion V39.0** engine, it analyzes your brand style, reference designs, and user prompts to generate complete poster compositions with background, subject, hero products, and marketing copy.

### The Idea

Traditional poster design requires significant design skills, time, and multiple tools. Our solution democratizes professional poster creation by:

1. **Intelligent Brand Analysis**: Upload a reference image, and our AI analyzes brand colors, style, and design patterns
2. **Context-Aware Generation**: The AI understands use cases (festivals, product launches, sales) and generates appropriate content
3. **Complete Poster Composition**: Automatically generates background, subject, product placement, and marketing copy
4. **Layer-Based Editing**: View and edit individual layers (background, subject, hero, logo) with custom text overlays

## ✨ Key Features

### 🎯 Core Functionality

- **AI-Powered Poster Generation**
  - Upload reference images to understand brand style
  - Generate complete marketing posters with a single prompt
  - Automatic detection of festivals, events, and product launches
  - Context-aware marketing copy generation (headlines, subheadlines, CTAs)

- **Brand Asset Integration**
  - Logo upload with automatic background removal
  - Reference image analysis for color extraction and style matching
  - Optional product image for product launch campaigns
  - Brand-aware color palette detection

- **Layer Editor & Text Customization**
  - View disintegrated poster layers (background, subject, hero, logo)
  - Toggle layer visibility
  - Add custom text with full control:
    - Font family selection (Arial, Helvetica, Times New Roman, etc.)
    - Font size (12-120px)
    - Text color (color picker + hex input)
    - Position control (X, Y coordinates)
  - Real-time canvas preview
  - Export edited posters

- **Smart Context Detection**
  - Festival detection (Diwali, Christmas, Holi, etc.)
  - Product launch vs. sale vs. general campaign
  - Automatic product suggestion based on context
  - Event-appropriate design elements

## 🏗️ Architecture

### Frontend (Adobe Express Add-on)
- **Technology**: HTML5, CSS3, JavaScript (Vanilla)
- **Framework**: Adobe CC Web Add-on SDK
- **Location**: `src/` directory

**Key Components:**
- `index.html` - Main UI structure
- `index.js` - Frontend logic and API integration
- `styles.css` - Modern dark theme styling
- `manifest.json` - Add-on configuration

### Backend (FastAPI)
- **Technology**: Python 3.8+, FastAPI
- **Location**: `backend/` directory

**Key Components:**
- `main.py` - FastAPI application with REST endpoints
- `branddiffusion.py` - BrandDiffusion V39.0 core engine (1874+ lines)
- `branddiffusion_wrapper.py` - API integration wrapper
- `branddiffusion_api.py` - Helper functions for image conversion

### AI Engine: BrandDiffusion V39.0

**Core Capabilities:**
- Stable Diffusion XL for image generation
- ControlNet for style consistency
- CLIP for product detection
- MediaPipe for face/hand detection
- GFPGAN for face restoration
- PosterLlama for layout generation
- Groq API for intelligent copy generation

## 📦 Installation & Setup

### Prerequisites

- **Node.js** (v14 or higher)
- **Python** 3.8 or higher
- **Adobe Express** account
- **SSL Certificate** (for local development)

### Frontend Setup

1. **Install Dependencies**
   ```bash
   npm install
   ```

2. **Setup SSL Certificate** (Required for Adobe CC add-ons)
   ```bash
   npx @adobe/ccweb-add-on-ssl setup --hostname localhost
   ```

   **Note**: If you encounter OpenSSL errors on Windows:
   - Install OpenSSL: `winget install ShiningLight.OpenSSL.Light`
   - Add to PATH: `C:\Program Files\OpenSSL-Win64\bin`

3. **Start Development Server**
   ```bash
   npm run start
   ```

   The add-on will be available at `https://localhost:3000` (or the configured SSL port)

### Backend Setup

1. **Create Virtual Environment**
   ```bash
   cd backend
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # macOS/Linux
   source venv/bin/activate
   ```

2. **Install Core Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install BrandDiffusion ML Dependencies** (Optional - Heavy)
   ```bash
   pip install -r requirements-branddiffusion.txt
   ```
   
   **Note**: These are large ML libraries (~5GB+). Install only if using full BrandDiffusion features.

4. **Environment Variables**
   Create a `.env` file in the `backend/` directory:
   ```env
   # Required for AI copy generation
   GROQ_API_KEY=your_groq_api_key_here
   
   # Required for logo background removal
   REMOVEBG_API_KEY=your_removebg_api_key_here
   
   # Optional - for Replicate integration
   REPLICATE_API_TOKEN=your_replicate_token_here
   ```

5. **Start Backend Server**
   ```bash
   python main.py
   ```
   
   The API will be available at `http://localhost:8000`
   - API Docs: `http://localhost:8000/docs`
   - Health Check: `http://localhost:8000/api/health`

## 🚀 Usage Guide

### Basic Workflow

1. **Upload Brand Assets**
   - Upload your logo (PNG, JPG, SVG - max 5MB)
   - Upload a reference image (PNG, JPG - max 5MB) - **Required**
   - Optionally upload a product image for product launch campaigns

2. **Describe Your Poster**
   - Enter a prompt describing what you want to create
   - Examples:
     - "Happy Diwali wishes for our customers with festive decorations"
     - "50% off summer collection sale"
     - "Launching new iPhone 15 Pro"

3. **Generate Poster**
   - Click "Generate Poster"
   - Wait for AI generation (typically 30-60 seconds)
   - View the generated poster with layers and marketing copy

4. **Edit Layers (Optional)**
   - Click "🎨 Edit Layers & Add Text" button
   - Toggle layer visibility
   - Add custom text with your preferred font, size, and color
   - Export your edited poster

5. **Add to Document**
   - Click "Add to Document" to import into Adobe Express
   - Continue editing in Adobe Express if needed

### Advanced Features

#### Layer Editor
- **View Individual Layers**: Background, Subject, Hero Product, Logo
- **Toggle Visibility**: Turn layers on/off to see individual components
- **Text Overlays**: Add multiple text layers with custom styling
- **Position Control**: Precise X, Y positioning for text elements
- **Export**: Save your customized poster as base64 image

#### Context-Aware Generation
The AI automatically detects:
- **Festivals**: Diwali, Christmas, Holi, Eid, etc.
- **Events**: Product launches, sales, general campaigns
- **Product Types**: Shoes, electronics, clothing, etc.
- **Design Style**: Based on reference image analysis

## 📡 API Endpoints

### `POST /api/generate-poster`
Generate a marketing poster using BrandDiffusion.

**Request:**
```json
{
  "prompt": "Happy Diwali wishes for our customers",
  "referenceImages": ["data:image/png;base64,..."],
  "logo": "data:image/png;base64,..." (optional),
  "productImage": "data:image/png;base64,..." (optional)
}
```

**Response:**
```json
{
  "success": true,
  "message": "Poster generated successfully",
  "composite": "data:image/png;base64,...",
  "layers": {
    "background": {
      "name": "background",
      "data": "data:image/png;base64,...",
      "x": 0,
      "y": 0,
      "width": 1280,
      "height": 800
    },
    "subject": {...},
    "hero": {...},
    "logo": {...}
  },
  "metadata": {
    "generated_copy": {
      "headline": "HAPPY DIWALI",
      "subheadline": "Wishing you joy, light, and warm moments",
      "cta": "CELEBRATE"
    },
    "use_case": "festival",
    "is_event": true,
    "event_name": "diwali"
  }
}
```

### `GET /api/health`
Health check endpoint.

### `POST /api/test-generation`
Test endpoint for development.

## 🔧 Technical Details

### Dependencies

#### Frontend
- `@adobe/ccweb-add-on-scripts` - Adobe Express add-on development tools
- Vanilla JavaScript (no frameworks)

#### Backend Core
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pillow` - Image processing
- `numpy` - Numerical operations
- `opencv-python` - Computer vision
- `scikit-learn` - Color extraction (KMeans)
- `pydantic` - Data validation

#### BrandDiffusion (Optional)
- `torch` - PyTorch for ML models
- `diffusers` - Stable Diffusion pipelines
- `transformers` - CLIP, LLM models
- `mediapipe` - Face/hand detection
- `gfpgan` - Face restoration
- `groq` - LLM inference for copy generation
- `peft` - LoRA adapters (PosterLlama)
- `bitsandbytes` - 4-bit quantization

### Project Structure

```
brand-ai-generator/
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── branddiffusion.py       # BrandDiffusion V39.0 core engine
│   ├── branddiffusion_wrapper.py  # API wrapper
│   ├── branddiffusion_api.py   # Helper functions
│   ├── requirements.txt        # Core dependencies
│   └── requirements-branddiffusion.txt  # ML dependencies
├── src/
│   ├── index.html             # Main UI
│   ├── index.js               # Frontend logic
│   ├── styles.css             # Styling
│   └── manifest.json          # Add-on manifest
├── dist/                      # Built files
├── document-sandbox/          # Adobe Express sandbox
├── package.json               # Node.js config
└── README.md                  # This file
```

## 🎨 UI/UX Features

- **Modern Dark Theme**: Sleek dark blue/purple gradient interface
- **Intuitive Drag & Drop**: Easy file uploads
- **Real-time Preview**: See uploaded images immediately
- **Progress Indicators**: Visual feedback during generation
- **Error Handling**: Clear error messages and validation
- **Responsive Design**: Works on different screen sizes

## 🧠 AI Features Explained

### BrandDiffusion V39.0 Engine

1. **Reference Analysis**
   - CLIP-based product detection
   - KMeans color extraction
   - Brightness analysis
   - Style pattern recognition

2. **Context Detection**
   - Groq API for intent analysis
   - Festival detection
   - Use case classification (festival/sale/launch/general)
   - Product prioritization

3. **Image Generation**
   - Stable Diffusion XL base model
   - ControlNet for style consistency
   - V9.4 context-aware hero generation
   - Face restoration (GFPGAN)
   - Hand inpainting

4. **Layout Generation**
   - PosterLlama layout engine (optional)
   - Fixed layout system (fallback)
   - Smart text positioning

5. **Copy Generation**
   - Groq API (Llama 3.3 70B)
   - Context-aware headlines
   - Festival-appropriate messaging
   - Sale-focused CTAs

## 🚨 Known Limitations

- **Large Dependencies**: BrandDiffusion ML libraries are ~5GB+
- **Generation Time**: Poster generation takes 30-60 seconds
- **GPU Recommended**: For faster generation, GPU with CUDA support
- **API Keys Required**: Groq and Remove.bg API keys needed for full functionality
- **Memory Usage**: Requires significant RAM (8GB+ recommended)

## 🔐 Security Notes

- CORS is currently open (`allow_origins=["*"]`) - restrict in production
- SSL required for Adobe Express add-on (handled by development tools)
- API keys should be stored in `.env` file (never commit to git)

## 📝 License

This project is part of a hackathon/demo project. Check with project maintainers for licensing details.

## 🤝 Contributing

This is a hackathon project. For improvements, please contact the project maintainers.

## 📞 Support

For issues or questions:
- Check API docs at `http://localhost:8000/docs`
- Review browser console for frontend errors
- Check backend logs for API errors

## 🎯 Future Enhancements

- [ ] Real-time layer preview during editing
- [ ] Drag-and-drop text positioning
- [ ] Multiple poster templates
- [ ] Batch generation
- [ ] Export to multiple formats (PNG, PDF, SVG)
- [ ] Integration with Adobe Express templates
- [ ] Custom font upload
- [ ] Animation support for reels

---

**Built with ❤️ for Adobe Express Hackathon**

*Powered by BrandDiffusion V39.0 "Event Booster"*
