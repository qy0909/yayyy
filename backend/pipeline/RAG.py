"""
Complete RAG Pipeline for Multilingual Bot
===========================================
This pipeline handles user queries through:
1. Language detection
2. Query embedding
3. Vector search in Supabase
4. LLM-based summarization and translation
5. Formatted response with sources
"""

import os
import re
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# 🔧 CONFIGURABLE: CORE LIBRARIES AND MODELS
# ============================================================================
from sentence_transformers import SentenceTransformer
from langdetect import detect, detect_langs
from openai import OpenAI
from supabase import create_client, Client
import spacy

# 🔧 HACKATHON: Southeast Asian LLM Support (Hugging Face Transformers)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not installed. Install with: pip install transformers torch")

# 🔧 CLOUD: Google Gemini Support
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️  Gemini not installed. Install with: pip install google-generativeai")

# 🔧 CLOUD: Groq Support
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not installed. Install with: pip install groq")

# 🔧 CLOUD: Hugging Face Inference API
try:
    from huggingface_hub import InferenceClient
    HF_INFERENCE_AVAILABLE = True
except ImportError:
    HF_INFERENCE_AVAILABLE = False
    print("⚠️  HF Hub not installed. Install with: pip install huggingface_hub")

# 🔧 TRANSLATION: NLLB-200 for ASEAN Dialect Translation
try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
    NLLB_AVAILABLE = True
except ImportError:
    NLLB_AVAILABLE = False
    print("⚠️  Translation models not available. Install with: pip install transformers sentencepiece")

# ============================================================================
# 🔧 CONFIGURABLE: MODEL SELECTION
# Change these models based on your requirements
# ============================================================================

# Embedding Model - Used for converting text to vectors
# Options: 'distiluse-base-multilingual-cased-v2', 'all-MiniLM-L6-v2', 
#          'paraphrase-multilingual-mpnet-base-v2'
EMBEDDING_MODEL_NAME = 'distiluse-base-multilingual-cased-v2'

# SpaCy Language Models
# Options: 'en_core_web_sm', 'en_core_web_md', 'en_core_web_lg' (English)
#          'xx_ent_wiki_sm' (Multilingual)
SPACY_MODEL_EN = 'en_core_web_sm'
SPACY_MODEL_MULTILINGUAL = 'xx_ent_wiki_sm'

# ============================================================================
# 🔧 HACKATHON: LLM Provider Selection
# Options: 'hf_inference', 'gemini', 'groq', 'huggingface', 'openai'
# ============================================================================
# 🌟 RECOMMENDED: 'hf_inference' - SEA-LION cloud (NO DOWNLOAD, best for ASEAN!)
LLM_PROVIDER = 'hf_inference'  # Change based on your preference

# Enable automatic fallback if primary provider fails
ENABLE_FALLBACK = True
FALLBACK_ORDER = ['hf_inference', 'gemini', 'groq', 'huggingface']  # Try in this order

# OpenAI LLM Model Selection
# Options: 'gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo', 'gpt-4o'
OPENAI_MODEL_NAME = 'gpt-3.5-turbo'

# ============================================================================
# 🔧 HACKATHON: Southeast Asian LLM Models (Hugging Face)
# Recommended for multilingual support (Tagalog, Malay, Thai, Vietnamese)
# ============================================================================
HUGGINGFACE_MODEL_OPTIONS = {
    # 🇸🇬 SEA-LION v4 - Best overall SEA language support
    'sealion': 'aisingapore/Apertus-SEA-LION-v4-8B-IT',
    
    # 🇸🇬 Sailor 2 - Good multilingual chat (lighter)
    'sailor': 'sail/Sailor2-8B-Chat',
    
    # 🇲🇾 ILMU - Malaysian-focused (Malay + English)
    'ilmu': 'malaysian-open-llm/ILMU-7B',
    
    # 🇹🇭 Typhoon - Thai + English
    'typhoon': 'scb10x/typhoon-7b-instruct',
    
    # 🇻🇳 VinaLLaMA - Vietnamese
    'vinallama': 'vilm/vinallama-7b-chat',
}

# Select which SEA model to use (change this!)
SEA_MODEL_CHOICE = 'sealion'  # Options: 'sealion', 'sailor', 'ilmu', 'typhoon', 'vinallama'
HUGGINGFACE_MODEL_NAME = HUGGINGFACE_MODEL_OPTIONS[SEA_MODEL_CHOICE]

# Use 4-bit quantization to reduce memory (8GB RAM → 4GB RAM)
USE_QUANTIZATION = True  # Set False if you have >16GB RAM

# ============================================================================
# 🔧 CLOUD: Model Selection for Cloud Providers
# ============================================================================

# Google Gemini Models
# 'gemini-1.5-flash' - Fastest, best for demos (FREE)
# 'gemini-1.5-pro' - Best quality, slower (FREE with limits)
# 'gemini-2.0-flash-exp' - Experimental, very fast
GEMINI_MODEL = 'gemini-1.5-flash'

# Groq Models (all FREE)
# 'llama-3.3-70b-versatile' - Newest, best quality
# 'llama-3.1-70b-versatile' - Stable, good performance
# 'mixtral-8x7b-32768' - Alternative, large context
GROQ_MODEL = 'llama-3.3-70b-versatile'

# Hugging Face Inference API Models
# Can use any public model on HF Hub
HF_INFERENCE_MODEL = 'aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct'

# ============================================================================
# 🔧 CONFIGURABLE: API KEYS AND DATABASE CONNECTION
# ============================================================================

# Set your API keys here or via environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')
SUPABASE_URL = os.getenv('SUPABASE_URL', 'YOUR_SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY', 'YOUR_SUPABASE_KEY')

# 🔧 CLOUD: API Keys for Free Cloud Providers
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')  # Get at: https://makersuite.google.com/app/apikey
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'YOUR_GROQ_API_KEY')  # Get at: https://console.groq.com
HF_TOKEN = os.getenv('HF_TOKEN', 'YOUR_HF_TOKEN')  # Get at: https://huggingface.co/settings/tokens

# ============================================================================
# 🔧 CONFIGURABLE: VECTOR SEARCH PARAMETERS
# ============================================================================

# Number of similar documents to retrieve
TOP_K_RESULTS = 1

# Supabase table name for embeddings
EMBEDDINGS_TABLE_NAME = 'embeddings'

# Similarity threshold (0-1, higher = more strict)
SIMILARITY_THRESHOLD = 0.15

# ============================================================================
# 🔧 CONFIGURABLE: LLM PROMPT SETTINGS
# ============================================================================

# Number of bullet points in summary
NUM_BULLET_POINTS = '3-5'

# Reading level for simplification
READING_LEVEL = '5th-grade'

# Reduce chunk size before sending retrieved context to the LLM
CHUNK_SUMMARY_MAX_CHARS = 900
CHUNK_SUMMARY_MAX_SENTENCES = 4
TOTAL_CONTEXT_MAX_CHARS = 3200

# Maximum tokens for LLM response
MAX_TOKENS = 1000

# Temperature for LLM (0-2, lower = more focused)
LLM_TEMPERATURE = 0.7

# ============================================================================
# 🔧 CONFIGURABLE: TRANSLATION SETTINGS (ASEAN Dialect Support)
# ============================================================================

# Enable dialect-aware translation
ENABLE_TRANSLATION = True

# Translation model - NLLB-200 supports 200+ languages including ASEAN dialects
# Options: 'facebook/nllb-200-distilled-600M' (faster, 2.4GB)
#          'facebook/nllb-200-1.3B' (better quality, 5GB)
#          'facebook/nllb-200-3.3B' (best quality, 13GB - not recommended for demo)
TRANSLATION_MODEL_NAME = 'facebook/nllb-200-distilled-600M'

# Pivot languages for RAG retrieval (your embeddings are in these languages)
PIVOT_LANGUAGES = ['eng_Latn', 'zsm_Latn']  # English and Malay

# NLLB Language Code Mapping for ASEAN Countries
# Format: 'langdetect_code' -> 'nllb_flores_code'
ASEAN_LANGUAGE_MAP = {
    # English
    'en': 'eng_Latn',
    
    # Malay/Malaysian/Indonesian
    'ms': 'zsm_Latn',      # Standard Malay
    'id': 'ind_Latn',      # Indonesian
    
    # Tagalog/Filipino
    'tl': 'tgl_Latn',      # Tagalog
    'fil': 'tgl_Latn',     # Filipino (use Tagalog)
    
    # Thai
    'th': 'tha_Thai',
    
    # Vietnamese
    'vi': 'vie_Latn',
    
    # Burmese/Myanmar
    'my': 'mya_Mymr',
    
    # Khmer/Cambodian
    'km': 'khm_Khmr',
    
    # Lao
    'lo': 'lao_Laoo',
    
    # Chinese variants
    'zh-cn': 'zho_Hans',   # Simplified Chinese
    'zh-tw': 'zho_Hant',   # Traditional Chinese
    'zh': 'zho_Hans',      # Default to Simplified
    
    # Tamil (Singapore/Malaysia)
    'ta': 'tam_Taml',
    
    # Javanese (Indonesia)
    'jv': 'jav_Latn',
    
    # Sundanese (Indonesia)
    'su': 'sun_Latn',
    
    # Cebuano (Philippines)
    'ceb': 'ceb_Latn',
    
    # Shan (Myanmar)
    'shn': 'shn_Mymr',
    
    # Minangkabau (Indonesia)
    'min': 'min_Latn',
    
    # Acehnese (Indonesia)
    'ace': 'ace_Latn',
    
    # Banjar (Indonesia)
    'bjn': 'bjn_Latn',
}

# Fallback behavior for unsupported dialects
TRANSLATION_FALLBACK = 'zsm_Latn'  # Default to Malay if dialect unsupported


class RAGPipeline:
    """
    Complete RAG (Retrieval-Augmented Generation) Pipeline
    """
    
    def __init__(self):
        """Initialize all models and connections"""
        print("🚀 Initializing RAG Pipeline...")
        
        # Debug logging for frontend
        self.debug_logs = []
        
        # Step 0: Initialize embedding model
        print(f"📦 Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"   ✓ Embedding dimension: {embedding_dim}")
        
        # Initialize SpaCy models
        print(f"📦 Loading SpaCy models...")
        try:
            self.nlp_en = spacy.load(SPACY_MODEL_EN)
            self.nlp_multi = spacy.load(SPACY_MODEL_MULTILINGUAL)
        except OSError as e:
            print(f"⚠️  SpaCy model not found: {e}")
            print("Run: python -m spacy download en_core_web_sm")
            print("Run: python -m spacy download xx_ent_wiki_sm")
            raise
        
        # Initialize Translation Model (NLLB-200 via Hugging Face Inference API)
        self.translator = None
        self.translation_enabled = False
        if ENABLE_TRANSLATION:
            if not HF_INFERENCE_AVAILABLE:
                print(f"⚠️  Translation enabled but huggingface_hub not available")
                print(f"   Install with: pip install huggingface_hub")
            elif not HF_TOKEN or HF_TOKEN == 'YOUR_HF_TOKEN':
                print(f"⚠️  Translation enabled but HF_TOKEN is not configured")
                print(f"   Get a token at: https://huggingface.co/settings/tokens")
            else:
                print(f"🌐 Initializing NLLB-200 translation via HF Inference API: {TRANSLATION_MODEL_NAME}")
                try:
                    # Use cloud-hosted NLLB model instead of downloading locally
                    self.translation_client = InferenceClient(
                        model=TRANSLATION_MODEL_NAME,
                        token=HF_TOKEN,
                    )
                    print(f"   ✓ Translation via HF Inference initialized "
                          f"(Supports {len(ASEAN_LANGUAGE_MAP)} ASEAN languages/dialects)")
                    self.translation_enabled = True
                except Exception as e:
                    print(f"   ⚠️  Translation client initialization failed: {e}")
                    print(f"   ℹ️  Continuing without translation support")
        
        # Initialize LLM based on provider
        self.llm_provider = LLM_PROVIDER
        self._init_llm_provider(LLM_PROVIDER)
    
    def _init_llm_provider(self, provider: str):
        """Initialize the specified LLM provider"""
        
        if provider == 'gemini' and GEMINI_AVAILABLE:
            print(f"🌟 Initializing Google Gemini: {GEMINI_MODEL}")
            genai.configure(api_key=GEMINI_API_KEY)
            self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)
            print(f"   ✓ Gemini initialized (Best for ASEAN multilingual!)")
            
        elif provider == 'groq' and GROQ_AVAILABLE:
            print(f"⚡ Initializing Groq: {GROQ_MODEL}")
            self.groq_client = Groq(api_key=GROQ_API_KEY)
            print(f"   ✓ Groq initialized (Ultra-fast inference!)")
            
        elif provider == 'hf_inference' and HF_INFERENCE_AVAILABLE:
            print(f"🤗 Initializing Hugging Face Inference API: {HF_INFERENCE_MODEL}")
            self.hf_inference_client = InferenceClient(token=HF_TOKEN)
            print(f"   ✓ HF Inference initialized (SEA-LION cloud access!)")
            
        elif provider == 'openai':
            print(f"🤖 Initializing OpenAI client with model: {OPENAI_MODEL_NAME}")
            self.openai_client = OpenAI(api_key=OPENAI_API_KEY)
            print(f"   ✓ OpenAI initialized")
            
        elif provider == 'huggingface' and TRANSFORMERS_AVAILABLE:
            print(f"💻 Loading Local SEA LLM: {SEA_MODEL_CHOICE} ({HUGGINGFACE_MODEL_NAME})")
            print(f"   ⚠️  This may take 2-5 minutes on first run (downloading model)...")
            
            if USE_QUANTIZATION:
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16
                )
                self.llm_model = pipeline(
                    "text-generation",
                    model=HUGGINGFACE_MODEL_NAME,
                    model_kwargs={"quantization_config": quantization_config},
                    device_map="auto"
                )
            else:
                self.llm_model = pipeline(
                    "text-generation",
                    model=HUGGINGFACE_MODEL_NAME,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
            print(f"   ✓ Local model loaded successfully!")
            
        else:
            available = []
            if GEMINI_AVAILABLE: available.append('gemini')
            if GROQ_AVAILABLE: available.append('groq')
            if HF_INFERENCE_AVAILABLE: available.append('hf_inference')
            if TRANSFORMERS_AVAILABLE: available.append('huggingface')
            available.append('openai')
            
            raise ValueError(
                f"Provider '{provider}' not available or dependencies not installed.\n"
                f"Available providers: {', '.join(available)}\n"
                f"Install missing: pip install google-generativeai groq huggingface_hub"
            )
        
        # Initialize Supabase client
        print(f"🗄️  Connecting to Supabase...")
        self.supabase_client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        print("✅ RAG Pipeline initialized successfully!\n")
    
    # ========================================================================
    # STEP 1: LANGUAGE DETECTION
    # ========================================================================
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """
        Detect the language and dialect of the input text
        
        Args:
            text: Input query text
            
        Returns:
            Dictionary with language code and confidence
        """
        print(f"🔍 Step 1: Detecting language for query: '{text[:50]}...'")
        
        try:
            # Get primary language
            primary_lang = detect(text)
            
            # Get all detected languages with probabilities
            lang_probs = detect_langs(text)
            
            result = {
                'primary_language': primary_lang,
                'all_languages': [
                    {'lang': lp.lang, 'probability': lp.prob} 
                    for lp in lang_probs
                ],
                'confidence': lang_probs[0].prob if lang_probs else 0.0
            }
            
            print(f"   ✓ Detected language: {primary_lang} (confidence: {result['confidence']:.2f})")
            return result
            
        except Exception as e:
            print(f"   ⚠️  Language detection failed: {e}")
            return {
                'primary_language': 'en',
                'all_languages': [{'lang': 'en', 'probability': 1.0}],
                'confidence': 0.0
            }
    
    # ========================================================================
    # STEP 1.5: DIALECT-AWARE TRANSLATION (ASEAN Languages)
    # ========================================================================
    
    def _is_embedding_language_supported(self, langdetect_code: str) -> bool:
        """
        Check if the language is reasonably covered by the embedding model.
        
        This roughly follows BGE-M3 / multilingual sentence-transformer coverage:
        - Major languages: English, Malay, Indonesian, Tagalog, Thai, Vietnamese,
          Burmese, Khmer, Lao, Chinese variants, etc.
        - Anything outside this set is treated as a potential low-resource dialect.
        """
        major_langs = {
            # Core pivot / national languages
            'en', 'ms', 'id',
            # ASEAN national languages
            'tl', 'fil', 'th', 'vi', 'my', 'km', 'lo',
            # Chinese variants (Singapore/Malaysia/region)
            'zh', 'zh-cn', 'zh-tw',
            # Some common regional / migrant languages
            'ta', 'jv', 'su', 'ceb', 'shn', 'min', 'ace', 'bjn',
        }
        return langdetect_code in major_langs
    
    def _normalize_language_hint(self, code: Optional[str]) -> Optional[str]:
        """
        Normalize various language codes from the frontend into the compact
        forms we use internally (e.g., 'ms-MY' -> 'ms', 'en-US' -> 'en').
        """
        if not code:
            return None
        
        c = code.strip().lower()
        
        direct_map = {
            'ms-my': 'ms',
            'en-us': 'en',
            'en-gb': 'en',
            'zh-cn': 'zh',
            'zh-hans': 'zh',
            'zh-tw': 'zh',
            'zh-hant': 'zh',
        }
        if c in direct_map:
            return direct_map[c]
        
        if '-' in c:
            base, _region = c.split('-', 1)
            return base
        
        return c
    
    def _detect_script_language_hint(self, text: str) -> Optional[str]:
        """
        Lightweight script-based heuristic to improve language detection accuracy,
        especially when langdetect is noisy.
        
        This never overrides an explicit user hint, but can override langdetect
        when script is very clear (e.g., Chinese, Thai, Khmer).
        """
        for ch in text:
            cp = ord(ch)
            # CJK Unified Ideographs → Chinese (generic)
            if 0x4E00 <= cp <= 0x9FFF:
                return 'zh'
            # Thai
            if 0x0E00 <= cp <= 0x0E7F:
                return 'th'
            # Lao
            if 0x0E80 <= cp <= 0x0EFF:
                return 'lo'
            # Khmer
            if 0x1780 <= cp <= 0x17FF:
                return 'km'
            # Myanmar
            if 0x1000 <= cp <= 0x109F:
                return 'my'
        return None
    
    def _get_nllb_language_code(self, langdetect_code: str) -> str:
        """
        Convert langdetect language code to NLLB-200 FLORES code
        
        Args:
            langdetect_code: Language code from langdetect (e.g., 'en', 'ms', 'tl')
            
        Returns:
            NLLB FLORES code (e.g., 'eng_Latn', 'zsm_Latn', 'tgl_Latn')
        """
        # Look up in our ASEAN language map
        nllb_code = ASEAN_LANGUAGE_MAP.get(langdetect_code, None)
        
        if nllb_code:
            return nllb_code
        
        # Fallback: use Malay as default for unsupported dialects
        self._log_debug(f"[Translation] ⚠️  Language '{langdetect_code}' not in ASEAN map, using fallback: {TRANSLATION_FALLBACK}")
        return TRANSLATION_FALLBACK
    
    def _needs_translation(self, nllb_code: str) -> bool:
        """Check if text needs translation to pivot language (Malay/English)"""
        return nllb_code not in PIVOT_LANGUAGES
    
    def translate_text(
        self,
        text: str,
        source_lang: str,
        target_lang: str
    ) -> Dict[str, Any]:
        """
        Translate text between ASEAN languages using NLLB-200
        
        Args:
            text: Text to translate
            source_lang: Source language (langdetect code or NLLB code)
            target_lang: Target language (langdetect code or NLLB code)
            
        Returns:
            Dictionary with translated text and metadata
        """
        # If translation is disabled, return original text
        if not self.translation_enabled:
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'translation_performed': False,
                'note': 'Translation disabled or model not available'
            }
        
        try:
            # Convert to NLLB codes if needed
            src_nllb = self._get_nllb_language_code(source_lang) if '_' not in source_lang else source_lang
            tgt_nllb = self._get_nllb_language_code(target_lang) if '_' not in target_lang else target_lang
            
            # Skip translation if source and target are the same
            if src_nllb == tgt_nllb:
                return {
                    'translated_text': text,
                    'source_lang': src_nllb,
                    'target_lang': tgt_nllb,
                    'translation_performed': False,
                    'note': 'Source and target language are the same'
                }
            
            self._log_debug(f"[Translation] 🌐 Translating: {src_nllb} → {tgt_nllb}")
            
            # Use Hugging Face Inference API NLLB-200 translation endpoint
            translated_text = self.translation_client.translation(
                text,
                src_lang=src_nllb,
                tgt_lang=tgt_nllb,
            )
            
            self._log_debug(f"[Translation] ✓ Translation completed")
            
            return {
                'translated_text': translated_text,
                'source_lang': src_nllb,
                'target_lang': tgt_nllb,
                'translation_performed': True,
                'original_text': text
            }
            
        except Exception as e:
            error_msg = f"Translation failed: {str(e)}"
            self._log_debug(f"[Translation] ⚠️  {error_msg}")
            print(f"   ⚠️  {error_msg}")
            
            return {
                'translated_text': text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'translation_performed': False,
                'error': error_msg,
                'note': 'Using original text due to translation error'
            }
    
    # ========================================================================
    # STEP 2: QUERY EMBEDDING
    # ========================================================================
    
    def create_query_embedding(self, query: str) -> List[float]:
        """
        Convert user query to vector embedding
        
        Args:
            query: User query text
            
        Returns:
            List of floats representing the embedding vector
        """
        print(f"🔢 Step 2: Creating query embedding...")
        
        # Generate embedding using the configured model
        embedding = self.embedding_model.encode(query)
        
        print(f"   ✓ Generated embedding of dimension: {len(embedding)}")
        return embedding.tolist()
    
    # ========================================================================
    # STEP 3: VECTOR SEARCH IN SUPABASE
    # ========================================================================
    
    def vector_search(self, query_embedding: List[float]) -> List[Dict[str, Any]]:
        """
        Search Supabase vector database for similar documents
        
        Args:
            query_embedding: Vector representation of the query
            
        Returns:
            List of relevant document chunks with metadata
        """
        print(f"🔎 Step 3: Searching vector database (top {TOP_K_RESULTS} results)...")
        
        try:
            # ================================================================
            # 🔧 CONFIGURABLE: Supabase Vector Search Query
            # Adjust this query based on your Supabase table structure
            # ================================================================
            
            # Example using Supabase's vector similarity search
            # Note: Adjust column names based on your schema
            response = self.supabase_client.rpc(
                'match_documents',  # Your Supabase function name
                {
                    'query_embedding': query_embedding,
                    'match_count': TOP_K_RESULTS,
                    'similarity_threshold': SIMILARITY_THRESHOLD
                }
            ).execute()
            
            results = response.data if response.data else []
            
            # Alternative: Direct table query if you have a different setup
            # response = self.supabase_client.table(EMBEDDINGS_TABLE_NAME) \
            #     .select("*") \
            #     .limit(TOP_K_RESULTS) \
            #     .execute()
            
            print(f"   ✓ Retrieved {len(results)} relevant documents")
            return results
            
        except Exception as e:
            print(f"   ⚠️  Vector search failed: {e}")
            print(f"   ℹ️  Make sure you have created the 'match_documents' function in Supabase")
            return []
    
    # ========================================================================
    # STEP 4: LLM PROMPT PREPARATION
    # ========================================================================
    
    def prepare_llm_prompt(
        self, 
        user_query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        target_language: str
    ) -> str:
        """
        Prepare the prompt for the LLM with retrieved context
        
        Args:
            user_query: Original user query
            retrieved_chunks: Relevant document chunks from vector search
            target_language: Language to translate the answer to
            
        Returns:
            Formatted prompt string
        """
        print(f"📝 Step 4: Preparing LLM prompt...")

        compressed_chunks = self._compress_retrieved_chunks(user_query, retrieved_chunks)
        
        # Format retrieved documents
        context_docs = []
        for idx, chunk in enumerate(compressed_chunks, 1):
            # Adjust field names based on your Supabase schema
            text = chunk.get('summary', chunk.get('content', chunk.get('text', '')))
            source = chunk.get('source_url', chunk.get('url', 'N/A'))
            title = chunk.get('title', 'Document')
            lang = chunk.get('language', 'unknown')
            
            context_docs.append(
                f"{idx}. [{title}] ({lang})\n"
                f"   Content: {text}\n"
                f"   Source: {source}\n"
            )
        
        context_text = "\n".join(context_docs)
        
        # ====================================================================
        # 🔧 CONFIGURABLE: LLM PROMPT TEMPLATE
        # Customize this prompt based on your specific requirements
        # ====================================================================
        
        prompt = f"""You are a helpful multilingual assistant for migrant workers and immigrants.

User Query (in {target_language}): "{user_query}"

Retrieved Documents:
{context_text}

Instructions:
1. Summarize the information into {NUM_BULLET_POINTS} clear bullet points
2. Simplify complex legal, medical, or technical jargon to {READING_LEVEL} reading level
3. Translate your entire answer to {target_language}
4. Include source URLs for each main point (use the format: [Source: URL])
5. If the documents don't contain relevant information, politely say so
6. Be accurate and helpful

Format your response as bullet points with sources.
"""
        
        print(f"   ✓ Prompt prepared ({len(prompt)} characters)")
        return prompt

    def _compress_retrieved_chunks(
        self,
        user_query: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Shrink retrieved chunks before prompt construction to fit the LLM context window."""
        compressed_chunks = []
        total_chars = 0
        original_total_chars = 0

        for chunk in retrieved_chunks:
            text = chunk.get('content', chunk.get('text', '')) or ''
            original_total_chars += len(text)

            summary = self._summarize_chunk_text(user_query, text)

            if total_chars >= TOTAL_CONTEXT_MAX_CHARS:
                break

            remaining_chars = TOTAL_CONTEXT_MAX_CHARS - total_chars
            if len(summary) > remaining_chars:
                summary = summary[:remaining_chars].rsplit(' ', 1)[0].strip()
                if summary:
                    summary += '...'

            if not summary:
                continue

            compressed_chunk = dict(chunk)
            compressed_chunk['summary'] = summary
            compressed_chunks.append(compressed_chunk)
            total_chars += len(summary)

        self._log_debug(
            f"[Step 4] 📉 Context compressed from {original_total_chars} to {total_chars} chars across {len(compressed_chunks)} chunks"
        )
        return compressed_chunks

    def _summarize_chunk_text(self, user_query: str, text: str) -> str:
        """Create a compact extractive summary of a retrieved chunk using query relevance."""
        cleaned_text = re.sub(r'\s+', ' ', (text or '')).strip()
        if not cleaned_text:
            return ''

        if len(cleaned_text) <= CHUNK_SUMMARY_MAX_CHARS:
            return cleaned_text

        query_terms = {
            token
            for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", user_query.lower())
            if len(token) > 2
        }

        nlp = self.nlp_en if all(ord(ch) < 128 for ch in cleaned_text[:200]) else self.nlp_multi
        doc = nlp(cleaned_text)
        sentences = [sentence.text.strip() for sentence in doc.sents if sentence.text.strip()]

        if not sentences:
            return cleaned_text[:CHUNK_SUMMARY_MAX_CHARS].rsplit(' ', 1)[0].strip() + '...'

        scored_sentences = []
        for index, sentence in enumerate(sentences):
            sentence_terms = {
                token
                for token in re.findall(r"[A-Za-zÀ-ÿ0-9']+", sentence.lower())
                if len(token) > 2
            }
            overlap_score = len(query_terms & sentence_terms)
            position_bonus = max(0, 2 - min(index, 2))
            length_bonus = min(len(sentence) / 200, 1)
            score = overlap_score * 3 + position_bonus + length_bonus
            scored_sentences.append((score, index, sentence))

        selected = []
        current_chars = 0
        for _, index, sentence in sorted(scored_sentences, key=lambda item: (-item[0], item[1])):
            sentence_length = len(sentence)
            if selected and len(selected) >= CHUNK_SUMMARY_MAX_SENTENCES:
                break
            if current_chars + sentence_length > CHUNK_SUMMARY_MAX_CHARS and selected:
                continue
            selected.append((index, sentence))
            current_chars += sentence_length + 1
            if current_chars >= CHUNK_SUMMARY_MAX_CHARS:
                break

        if not selected:
            summary = cleaned_text[:CHUNK_SUMMARY_MAX_CHARS]
        else:
            summary = ' '.join(sentence for _, sentence in sorted(selected, key=lambda item: item[0]))

        summary = summary.strip()
        if len(summary) < len(cleaned_text):
            summary += '...'
        return summary
    
    # ========================================================================
    # STEP 5: CALL LLM
    # ========================================================================
    
    def call_llm(self, prompt: str) -> str:
        """
        Call the LLM to generate a response with automatic fallback
        
        Args:
            prompt: Prepared prompt with context
            
        Returns:
            LLM-generated response text
        """
        # Try primary provider
        try:
            return self._call_llm_provider(self.llm_provider, prompt)
        except Exception as e:
            error_msg = f"⚠️  Primary provider '{self.llm_provider}' failed: {str(e)}"
            self._log_debug(f"[Step 5] {error_msg}")
            print(f"   {error_msg}")
            
            # Try fallback providers if enabled
            if ENABLE_FALLBACK:
                for fallback_provider in FALLBACK_ORDER:
                    if fallback_provider == self.llm_provider:
                        continue  # Skip already tried provider
                    
                    try:
                        fallback_msg = f"🔄 Trying fallback provider: {fallback_provider}"
                        self._log_debug(f"[Step 5] {fallback_msg}")
                        print(f"   {fallback_msg}")
                        self._init_llm_provider(fallback_provider)
                        return self._call_llm_provider(fallback_provider, prompt)
                    except Exception as fallback_error:
                        fallback_error_msg = f"⚠️  Fallback '{fallback_provider}' also failed: {str(fallback_error)}"
                        self._log_debug(f"[Step 5] {fallback_error_msg}")
                        print(f"   {fallback_error_msg}")
                        continue
            
            # All providers failed
            final_error = f"❌ All LLM providers failed. Last error: {str(e)}"
            self._log_debug(f"[Step 5] {final_error}")
            return "Sorry, I couldn't generate a response at this time. Please check your API keys and internet connection."
    
    def _call_llm_provider(self, provider: str, prompt: str) -> str:
        """Route to specific provider's LLM call"""
        if provider == 'gemini':
            return self._call_gemini_llm(prompt)
        elif provider == 'groq':
            return self._call_groq_llm(prompt)
        elif provider == 'hf_inference':
            return self._call_hf_inference_llm(prompt)
        elif provider == 'openai':
            return self._call_openai_llm(prompt)
        elif provider == 'huggingface':
            return self._call_huggingface_llm(prompt)
        else:
            raise ValueError(f"Unknown provider: {provider}")
    
    def _call_gemini_llm(self, prompt: str) -> str:
        """Call Google Gemini API (Best for ASEAN multilingual)"""
        print(f"🌟 Step 5: Calling Google Gemini ({GEMINI_MODEL})...")
        self._log_debug(f"[Step 5] 🌟 Using Gemini: {GEMINI_MODEL}")
        
        try:
            # Check if API key is valid
            if not GEMINI_API_KEY or GEMINI_API_KEY == 'YOUR_GEMINI_API_KEY':
                error = "GEMINI_API_KEY not configured. Get your key at: https://makersuite.google.com/app/apikey"
                self._log_debug(f"[Step 5] ❌ {error}")
                raise ValueError(error)
            
            self._log_debug(f"[Step 5] 📡 Sending request to Gemini API...")
            
            # Generate response with Gemini
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    max_output_tokens=MAX_TOKENS,
                    temperature=LLM_TEMPERATURE,
                )
            )
            
            answer = response.text
            self._log_debug(f"[Step 5] ✓ Gemini response: {len(answer)} chars")
            print(f"   ✓ Gemini response generated ({len(answer)} characters)")
            print(f"   ℹ️  Provider: Google Gemini (Excellent ASEAN dialect support!)")
            return answer
            
        except Exception as e:
            error_details = f"Gemini API failed: {type(e).__name__}: {str(e)}"
            self._log_debug(f"[Step 5] ❌ {error_details}")
            print(f"   ⚠️  {error_details}")
            
            if "API_KEY" in str(e).upper() or "401" in str(e):
                self._log_debug(f"[Step 5] 💡 Fix: Check your GEMINI_API_KEY in .env file")
            
            raise
    
    def _call_groq_llm(self, prompt: str) -> str:
        """Call Groq API (Ultra-fast inference)"""
        print(f"⚡ Step 5: Calling Groq ({GROQ_MODEL})...")
        self._log_debug(f"[Step 5] ⚡ Using Groq: {GROQ_MODEL}")
        
        try:
            # Check if API key is valid
            if not GROQ_API_KEY or GROQ_API_KEY == 'YOUR_GROQ_API_KEY':
                error = "GROQ_API_KEY not configured. Get your key at: https://console.groq.com"
                self._log_debug(f"[Step 5] ❌ {error}")
                raise ValueError(error)
            
            self._log_debug(f"[Step 5] 📡 Sending request to Groq API...")
            
            response = self.groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant specializing in providing "
                                   "simplified information to migrant workers in Southeast Asia."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            
            answer = response.choices[0].message.content
            self._log_debug(f"[Step 5] ✓ Groq response: {len(answer)} chars, {response.usage.total_tokens} tokens")
            print(f"   ✓ Groq response generated ({len(answer)} characters)")
            print(f"   ℹ️  Speed: Ultra-fast! Tokens: {response.usage.total_tokens}")
            return answer
            
        except Exception as e:
            error_details = f"Groq API failed: {type(e).__name__}: {str(e)}"
            self._log_debug(f"[Step 5] ❌ {error_details}")
            print(f"   ⚠️  {error_details}")
            
            if "api_key" in str(e).lower() or "401" in str(e):
                self._log_debug(f"[Step 5] 💡 Fix: Check your GROQ_API_KEY in .env file")
            
            raise
    
    def _call_hf_inference_llm(self, prompt: str) -> str:
        """Call Hugging Face Inference API (Cloud SEA models)"""
        print(f"🤗 Step 5: Calling HF Inference ({HF_INFERENCE_MODEL.split('/')[-1]})...")
        self._log_debug(f"[Step 5] 🤗 Using HF Inference: {HF_INFERENCE_MODEL}")
        
        try:
            # Check if token is valid
            if not HF_TOKEN or HF_TOKEN == 'YOUR_HF_TOKEN':
                error = "HF_TOKEN not configured. Get your token at: https://huggingface.co/settings/tokens"
                self._log_debug(f"[Step 5] ❌ {error}")
                raise ValueError(error)
            
            # Format for chat models
            messages = [{
                "role": "user",
                "content": prompt
            }]
            
            self._log_debug(f"[Step 5] 📡 Sending request to HF Inference API...")
            
            response = self.hf_inference_client.chat_completion(
                messages=messages,
                model=HF_INFERENCE_MODEL,
                max_tokens=MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            
            answer = response.choices[0].message.content
            self._log_debug(f"[Step 5] ✓ HF Inference response: {len(answer)} chars")
            print(f"   ✓ HF Inference response generated ({len(answer)} characters)")
            print(f"   ℹ️  Using cloud-hosted SEA-LION model")
            return answer
            
        except Exception as e:
            error_details = f"HF Inference API failed: {type(e).__name__}: {str(e)}"
            self._log_debug(f"[Step 5] ❌ {error_details}")
            print(f"   ⚠️  {error_details}")
            
            # Provide helpful error messages
            if "401" in str(e) or "unauthorized" in str(e).lower():
                self._log_debug(f"[Step 5] 💡 Fix: Check your HF_TOKEN in .env file")
            elif "model" in str(e).lower():
                self._log_debug(f"[Step 5] 💡 Fix: Model {HF_INFERENCE_MODEL} might not support Inference API")
            elif "timeout" in str(e).lower():
                self._log_debug(f"[Step 5] 💡 Fix: Check your internet connection")
            
            raise
    
    def _call_openai_llm(self, prompt: str) -> str:
        """Call OpenAI GPT models"""
        print(f"🤖 Step 5: Calling OpenAI LLM ({OPENAI_MODEL_NAME})...")
        
        try:
            response = self.openai_client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a helpful assistant specializing in providing "
                                   "simplified information to migrant workers."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            answer = response.choices[0].message.content
            print(f"   ✓ LLM response generated ({len(answer)} characters)")
            print(f"   ℹ️  Tokens used: {response.usage.total_tokens}")
            return answer
            
        except Exception as e:
            print(f"   ⚠️  OpenAI LLM call failed: {e}")
            return "Sorry, I couldn't generate a response at this time."
    
    def _call_huggingface_llm(self, prompt: str) -> str:
        """Call Hugging Face Southeast Asian LLM models"""
        print(f"🤖 Step 5: Calling SEA LLM ({SEA_MODEL_CHOICE})...")
        
        try:
            # Format prompt for chat models
            formatted_prompt = f"""<|system|>
You are a helpful assistant specializing in providing simplified information to migrant workers in Southeast Asia.
<|user|>
{prompt}
<|assistant|>
"""
            
            # Generate response
            response = self.llm_model(
                formatted_prompt,
                max_new_tokens=MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.llm_model.tokenizer.eos_token_id
            )
            
            # Extract answer (remove prompt)
            full_text = response[0]['generated_text']
            answer = full_text.split('<|assistant|>')[-1].strip()
            
            print(f"   ✓ SEA LLM response generated ({len(answer)} characters)")
            return answer
            
        except Exception as e:
            print(f"   ⚠️  Hugging Face LLM call failed: {e}")
            return "Sorry, I couldn't generate a response at this time."
    
    # ========================================================================
    # STEP 6: POST-PROCESSING
    # ========================================================================
    
    def post_process_response(self, llm_response: str, language: str) -> Dict[str, Any]:
        """
        Format the LLM response for frontend display
        
        Args:
            llm_response: Raw response from LLM
            language: Detected/target language
            
        Returns:
            Formatted response dictionary
        """
        print(f"✨ Step 6: Post-processing response...")
        
        # Split response into bullet points
        lines = llm_response.strip().split('\n')
        bullet_points = []
        
        for line in lines:
            line = line.strip()
            if line and (line.startswith('•') or line.startswith('-') or 
                        line.startswith('*') or line[0].isdigit()):
                # Clean up bullet point
                cleaned = line.lstrip('•-*0123456789. ')
                if cleaned:
                    bullet_points.append(f"• {cleaned}")
        
        # If no bullet points found, treat entire response as one point
        if not bullet_points:
            bullet_points = [f"• {llm_response}"]
        
        result = {
            'answer': bullet_points,
            'language': language,
            'timestamp': datetime.now().isoformat(),
            'raw_response': llm_response
        }
        
        print(f"   ✓ Response formatted into {len(bullet_points)} bullet points")
        return result

    # ========================================================================
    # STEP 8: RECURSIVE SUMMARIZATION
    # ========================================================================

    def summarize_document(self, document_text: str, chunk_size: int = 4000) -> str:
        """
        Recursively summarize a long document into 3-5 bullet points.

        Args:
            document_text: The long text of the document to summarize.
            chunk_size: The approximate size of each chunk in characters.

        Returns:
            A string containing the final summary in bullet points.
        """
        self._log_debug(f"[Summarize] 🔄 Starting recursive summarization for document of {len(document_text)} chars.")

        # 1. Chunk the document
        chunks = self._chunk_text(document_text, chunk_size)
        self._log_debug(f"[Summarize] 📄 Split document into {len(chunks)} chunks.")

        # 2. Recursively summarize
        summary = self._recursive_summarize_chunks(chunks)
        self._log_debug(f"[Summarize] ✨ Final condensed summary: {len(summary)} chars.")

        # 3. Final formatting
        final_prompt = f"""Based on the following summary, generate a final list of 3-5 actionable bullet points.
The points should be concise and easy to understand.

Summary:
{summary}

Final bullet points:
"""
        final_summary = self.call_llm(final_prompt)
        self._log_debug(f"[Summarize] ✅ Final bullet points generated.")
        return final_summary

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Splits text into chunks of a specified size without breaking sentences."""
        if len(text) <= chunk_size:
            return [text]

        nlp = self.nlp_en if all(ord(ch) < 128 for ch in text[:200]) else self.nlp_multi
        doc = nlp(text)
        sentences = [s.text for s in doc.sents]
        
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) > chunk_size:
                chunks.append(current_chunk)
                current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks

    def _recursive_summarize_chunks(self, chunks: List[str]) -> str:
        """Helper for the recursive summarization logic."""
        if len(chunks) == 1:
            # Base case: summarize the final chunk
            prompt = f"Summarize the following text concisely:\n\n{chunks[0]}"
            summary = self.call_llm(prompt)
            return summary

        # Summarize each chunk individually
        summaries = []
        for i, chunk in enumerate(chunks):
            self._log_debug(f"[Summarize] summarizing chunk {i+1}/{len(chunks)}")
            prompt = f"Summarize the following text concisely:\n\n{chunk}"
            summary = self.call_llm(prompt)
            summaries.append(summary)

        # Recursive step: summarize the summaries
        concatenated_summaries = "\n\n".join(summaries)
        new_chunks = self._chunk_text(concatenated_summaries, 4000)
        return self._recursive_summarize_chunks(new_chunks)

    # ========================================================================
    # STEP 9: LEXICAL SIMPLIFICATION
    # ========================================================================

    def simplify_text(self, text: str, reading_level: str = '5th-grade') -> str:
        """
        Simplifies complex jargon in a text to a specified reading level.

        Args:
            text: The text to simplify.
            reading_level: The target reading level (e.g., '5th-grade', 'simple').

        Returns:
            The simplified text.
        """
        self._log_debug(f"[Simplify] 🔬 Simplifying text to {reading_level} level.")

        prompt = f"""Please simplify the following text to a {reading_level} reading level.
Replace complex legal, medical, or technical jargon with simple, everyday language.
Do not change the meaning of the text.

Original text:
{text}

Simplified text:
"""
        simplified_text = self.call_llm(prompt)
        self._log_debug(f"[Simplify] ✅ Text simplified.")
        return simplified_text
    
    # ========================================================================
    # STEP 7: MAIN PIPELINE EXECUTION
    # ========================================================================
    
    def _log_debug(self, message: str):
        """Add debug message to logs (for frontend display)"""
        self.debug_logs.append(message)
        print(message)  # Also print to terminal
    
    def process_query(self, user_query: str, user_language_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Main pipeline: Process a user query end-to-end with dialect-aware translation
        
        Workflow:
        1. Detect user's language/dialect (using user_language_hint if provided)
        2. Translate query to pivot language (Malay/English) if needed
        3. Create embedding from (translated) query
        4. RAG retrieval
        5. LLM generates answer
        6. Translate answer back to user's dialect
        7. Return response with sources
        
        Args:
            user_query: User's input query in any ASEAN dialect
            user_language_hint: Optional language code from frontend (e.g., 'ms', 'en', 'zh')
            
        Returns:
            Complete response with answer, sources, and metadata
        """
        # Clear debug logs for this query
        self.debug_logs = []
        
        self._log_debug("\n" + "="*70)
        self._log_debug(f"🎯 Processing Query: '{user_query}'")
        self._log_debug("="*70 + "\n")
        
        try:
            # Step 1: Detect language/dialect
            self._log_debug("[Step 1] 🔍 Detecting language...")
            
            # 1a. Use explicit user language hint if provided (highest priority)
            hint_normalized = self._normalize_language_hint(user_language_hint) if user_language_hint else None
            if hint_normalized:
                detected_language = hint_normalized
                self._log_debug(f"[Step 1] ℹ️ Using user-provided language hint: {detected_language}")
            else:
                # 1b. Try script-based heuristic
                script_hint = self._detect_script_language_hint(user_query)
                detected_language = script_hint if script_hint else None
                
                # 1c. Fallback to langdetect
                try:
                    lang_info = self.detect_language(user_query)
                    primary_lang = lang_info['primary_language']
                    self._log_debug(f"[Step 1] ℹ️ Langdetect primary: {primary_lang}")
                    
                    # If script-based hint exists and disagrees, prefer script hint for clear scripts
                    if detected_language and detected_language != primary_lang:
                        self._log_debug(
                            f"[Step 1] ⚖️ Script hint '{detected_language}' vs langdetect '{primary_lang}' "
                            f"→ preferring script hint"
                        )
                    else:
                        detected_language = primary_lang
                except Exception as e:
                    self._log_debug(f"[Step 1] ⚠️ Langdetect failed, defaulting to English: {e}")
                    if not detected_language:
                        detected_language = 'en'
            user_nllb_code = self._get_nllb_language_code(detected_language)
            self._log_debug(f"[Step 1] ✓ Detected: {detected_language} ({user_nllb_code})")
            
            # Check coverage: embeddings + translation
            embedding_supported = self._is_embedding_language_supported(detected_language)
            translation_supported = user_nllb_code in ASEAN_LANGUAGE_MAP.values()
            
            # True low-resource dialect: neither embeddings nor NLLB can reasonably handle it
            if not embedding_supported and (not self.translation_enabled or not translation_supported):
                self._log_debug("[Step 1] ⚠️ Detected potential low-resource dialect "
                                "(no embedding or translation coverage)")
                
                fallback_msg = (
                    "Sorry, your dialect is currently not supported. "
                    "Please enter your question in Malay or English."
                )
                
                return {
                    'answer': [f"• {fallback_msg}"],
                    'language': detected_language,
                    'user_language_code': user_nllb_code,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'low_resource_dialect',
                    'debug_logs': self.debug_logs,
                    'translation_note': (
                        "Low-resource dialect: neither embedding model nor NLLB-200 "
                        "cover this language reliably."
                    ),
                }
            
            # Step 1.5: Translate query to pivot language if needed
            query_for_retrieval = user_query
            translation_note = None
            
            if self.translation_enabled and self._needs_translation(user_nllb_code):
                self._log_debug(f"[Step 1.5] 🌐 Translating query to pivot language...")
                
                # Choose pivot language (prefer Malay for ASEAN, English as fallback)
                pivot_lang = 'zsm_Latn'  # Malay
                
                translation_result = self.translate_text(
                    text=user_query,
                    source_lang=user_nllb_code,
                    target_lang=pivot_lang
                )
                
                if translation_result['translation_performed']:
                    query_for_retrieval = translation_result['translated_text']
                    self._log_debug(f"[Step 1.5] ✓ Query translated: '{query_for_retrieval[:50]}...'")
                else:
                    self._log_debug(f"[Step 1.5] ⚠️  Translation skipped, using original query")
            else:
                if self.translation_enabled:
                    self._log_debug(f"[Step 1.5] ℹ️  Query already in pivot language, no translation needed")
                else:
                    self._log_debug(f"[Step 1.5] ⚠️  Translation disabled")
                    # Add disclaimer if translation is needed but disabled
                    if user_nllb_code not in PIVOT_LANGUAGES:
                        translation_note = f"⚠️ Note: Translation unavailable. Showing results in Malay/English."
            
            # Step 2: Create query embedding (from translated query)
            self._log_debug("[Step 2] 📊 Creating query embedding...")
            query_embedding = self.create_query_embedding(query_for_retrieval)
            self._log_debug(f"[Step 2] ✓ Embedding created (dim: {len(query_embedding)})")
            
            # Step 3: Vector search
            self._log_debug("[Step 3] 🔎 Searching vector database...")
            retrieved_chunks = self.vector_search(query_embedding)
            
            if not retrieved_chunks:
                self._log_debug("[Step 3] ⚠️ No relevant documents found")
                no_results_msg = "• No relevant information found for your query. Please try rephrasing."
                
                # Translate "no results" message back to user's language if needed
                if self.translation_enabled and self._needs_translation(user_nllb_code):
                    trans_result = self.translate_text(
                        text=no_results_msg,
                        source_lang='eng_Latn',
                        target_lang=user_nllb_code
                    )
                    if trans_result['translation_performed']:
                        no_results_msg = "• " + trans_result['translated_text'].lstrip('• ')
                
                return {
                    'answer': [no_results_msg],
                    'language': detected_language,
                    'user_language_code': user_nllb_code,
                    'timestamp': datetime.now().isoformat(),
                    'status': 'no_results',
                    'debug_logs': self.debug_logs,
                    'translation_note': translation_note
                }
            
            self._log_debug(f"[Step 3] ✓ Found {len(retrieved_chunks)} relevant chunks")
            
            # Step 4: Prepare prompt (use pivot language for LLM)
            self._log_debug("[Step 4] 📝 Preparing LLM prompt...")
            prompt = self.prepare_llm_prompt(
                query_for_retrieval,  # Use translated query
                retrieved_chunks, 
                'zsm_Latn' if user_nllb_code != 'eng_Latn' else 'eng_Latn'  # Ask LLM for Malay/English first
            )
            self._log_debug(f"[Step 4] ✓ Prompt prepared (length: {len(prompt)})")
            
            # Step 5: Call LLM
            self._log_debug(f"[Step 5] 🤖 Calling LLM ({self.llm_provider})...")
            llm_response = self.call_llm(prompt)
            self._log_debug(f"[Step 5] ✓ LLM response received (length: {len(llm_response)})")
            
            # Step 6: Post-process
            self._log_debug("[Step 6] 🔧 Post-processing response...")
            final_response = self.post_process_response(llm_response, detected_language)
            
            # Step 6.5: Translate answer back to user's dialect
            if self.translation_enabled and self._needs_translation(user_nllb_code):
                self._log_debug(f"[Step 6.5] 🌐 Translating answer back to user's language ({user_nllb_code})...")
                
                # Translate each bullet point
                translated_bullets = []
                for bullet in final_response['answer']:
                    # Remove bullet symbol before translation
                    text_only = bullet.lstrip('• -*').strip()
                    
                    trans_result = self.translate_text(
                        text=text_only,
                        source_lang='zsm_Latn',  # From Malay
                        target_lang=user_nllb_code
                    )
                    
                    if trans_result['translation_performed']:
                        translated_bullets.append(f"• {trans_result['translated_text']}")
                    else:
                        translated_bullets.append(bullet)  # Keep original if translation fails
                
                final_response['answer'] = translated_bullets
                final_response['answer_original_language'] = llm_response  # Keep original for reference
                self._log_debug(f"[Step 6.5] ✓ Answer translated to user's language")
            else:
                if not self.translation_enabled and user_nllb_code not in PIVOT_LANGUAGES:
                    # Add note that we're showing in Malay/English
                    final_response['answer'].insert(0, translation_note or "⚠️ Showing results in Malay/English (translation unavailable)")
            
            # Add metadata
            final_response['status'] = 'success'
            final_response['retrieved_chunks_count'] = len(retrieved_chunks)
            final_response['detected_language'] = detected_language
            final_response['user_language_code'] = user_nllb_code
            final_response['query_translated'] = query_for_retrieval != user_query
            final_response['answer_translated'] = self.translation_enabled and self._needs_translation(user_nllb_code)
            
            # Add sources for frontend display
            final_response['sources'] = retrieved_chunks
            
            # Add debug logs to response
            final_response['debug_logs'] = self.debug_logs
            
            self._log_debug("\n" + "="*70)
            self._log_debug("✅ Query processing completed successfully!")
            self._log_debug("="*70 + "\n")
            
            return final_response
            
        except Exception as e:
            self._log_debug(f"\n❌ Error processing query: {e}")
            import traceback
            self._log_debug(f"Traceback: {traceback.format_exc()}")
            
            return {
                'answer': [f"• Error: {str(e)}"],
                'language': 'en',
                'timestamp': datetime.now().isoformat(),
                'status': 'error',
                'debug_logs': self.debug_logs
            }


# ============================================================================
# STEP 8: OPTIONAL ENHANCEMENTS
# ============================================================================

class RAGPipelineEnhanced(RAGPipeline):
    """
    Enhanced RAG Pipeline with optional features:
    - Voice output (TTS)
    - Query logging
    - Fallback handling
    """
    
    def __init__(self, enable_logging: bool = True):
        super().__init__()
        self.enable_logging = enable_logging
        self.query_log = []
    
    def log_query(self, query: str, response: Dict[str, Any]):
        """Log query-answer pairs for demo/analysis"""
        if self.enable_logging:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'query': query,
                'response': response,
                'language': response.get('language', 'unknown')
            }
            self.query_log.append(log_entry)
    
    def save_logs(self, filepath: str = 'query_logs.json'):
        """Save logs to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.query_log, f, ensure_ascii=False, indent=2)
        print(f"📊 Logs saved to {filepath}")
    
    def process_query_with_logging(self, user_query: str) -> Dict[str, Any]:
        """Process query and log the interaction"""
        response = self.process_query(user_query)
        self.log_query(user_query, response)
        return response


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Initialize the pipeline
    rag = RAGPipeline()
    
    # Example query
    example_query = "Paano ako makakakuha ng healthcare subsidy sa Penang?"
    
    # Process the query
    result = rag.process_query(example_query)
    
    # Print result
    print("\n📋 FINAL RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
