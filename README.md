# SuaraGov - Inclusive Citizen AI

**No policy left unclear.** SuaraGov is a multilingual, RAG-powered public service AI assistant designed to help citizens understand government policies, eligibility requirements, and application procedures in plain, simple language.

## 🌟 Key Features

- **Multilingual & Dialect Support:** Automatically detects the user's language (supporting ASEAN languages like Malay, Tagalog, Thai, Vietnamese, Indonesian, etc.) and translates queries/responses using NLLB-200.
- **Verifiable RAG Pipeline:** Every answer is backed by official government documents. The UI highlights source excerpts and provides jump links to the original text.
- **Voice-First Accessibility:** Integrated Voice-to-Text (Whisper) for recording queries, and highly human-like Text-to-Speech (Microsoft Edge Neural TTS) for reading answers aloud.
- **Inclusive Community Dictionary:** Allows users to submit local slang or phrases to be reviewed by admins, improving the AI's understanding of underrepresented communities.
- **PDF Export:** Users can generate and download a beautifully formatted PDF of their chat history for offline reference.
- **Multi-User & Snappy UI:** Uses Next.js API routes with a session-based architecture to isolate user chats via Supabase, while utilizing `localStorage` Optimistic UI caching for zero-latency load times.

---

## 🛠 Tech Stack

**Frontend (Vercel-ready)**
- Next.js (React, TypeScript)
- Tailwind CSS
- jsPDF (for offline chat exports)
- React Markdown

**Backend (Render/Railway-ready)**
- Python & FastAPI
- Langdetect & SpaCy (NLP)
- Supabase (PostgreSQL + pgvector for Embeddings)
- Groq & Hugging Face Hub (LLM Inference & Translation)
- OpenAI Whisper & Edge-TTS (Audio processing)

---

## 🚀 Getting Started

### Prerequisites
- Node.js (v18+)
- Python (v3.10+)
- FFmpeg (Required for Whisper Voice-to-Text processing)
- A Supabase account and database
- Free API Keys: Groq and Hugging Face

### 1. Install System Dependencies (FFmpeg)
FFmpeg is required for the voice-to-text functionality. Choose the command for your OS:

- **macOS:** brew install ffmpeg

- **Linux:** sudo apt update && sudo apt install ffmpeg

- **Windows:** choco install ffmpeg (or download from ffmpeg.org)

### 2. Setup the Backend (FastAPI)

Navigate to the backend directory, set up your virtual environment, and install dependencies:

```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download required NLP models
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm
```

Create a `.env` file in the `backend/` folder (or project root):
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
SLANG_REVIEW_ADMIN_TOKEN=your_secret_admin_password
```

Start the Python server:
```bash
cd backend
python main.py
# Server runs on http://127.0.0.1:8000
```

### 3. Setup the Frontend (Next.js)

Open a new terminal and navigate to the root directory:

```bash
# Install Node modules
npm install
```

Create a `.env.local` file in the root directory:
```env
PYTHON_API_URL=http://127.0.0.1:8000
```

Start the development server:
```bash
npm run dev
# App runs on http://localhost:3000
```




