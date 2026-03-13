"""
Diagnostic script to test LLM API connections
Run this to see which LLM providers are working and which need configuration
"""

import os
from dotenv import load_dotenv

load_dotenv()

print("=" * 70)
print("🔍 LLM API Diagnostics")
print("=" * 70)
print()

# Check environment variables
print("📋 Checking Environment Variables:")
print("-" * 70)

HF_TOKEN = os.getenv('HF_TOKEN', 'YOUR_HF_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY', 'YOUR_GEMINI_API_KEY')
GROQ_API_KEY = os.getenv('GROQ_API_KEY', 'YOUR_GROQ_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'YOUR_OPENAI_API_KEY')

def check_key(name, value):
    if value.startswith('YOUR_') or not value or len(value) < 10:
        print(f"❌ {name}: NOT SET or INVALID")
        return False
    else:
        print(f"✅ {name}: Set ({value[:10]}...)")
        return True

hf_ok = check_key("HF_TOKEN", HF_TOKEN)
gemini_ok = check_key("GEMINI_API_KEY", GEMINI_API_KEY)
groq_ok = check_key("GROQ_API_KEY", GROQ_API_KEY)
openai_ok = check_key("OPENAI_API_KEY", OPENAI_API_KEY)

print()
print("=" * 70)
print("🧪 Testing API Connections:")
print("-" * 70)

# Test Hugging Face Inference API
if hf_ok:
    try:
        from huggingface_hub import InferenceClient
        print("\n🤗 Testing Hugging Face Inference API...")
        client = InferenceClient(token=HF_TOKEN)
        
        # Simple test
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello"}],
            model="aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct",
            max_tokens=50,
        )
        
        result = response.choices[0].message.content
        print(f"✅ HF Inference API: WORKING")
        print(f"   Response: {result[:100]}")
        
    except Exception as e:
        print(f"❌ HF Inference API: FAILED")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   💡 Fix: Check your HF_TOKEN at https://huggingface.co/settings/tokens")
        print(f"   💡 Make sure you have access to the model")
else:
    print("\n❌ HF_TOKEN not set - skipping test")
    print("   💡 Get token at: https://huggingface.co/settings/tokens")
    print("   💡 Add to .env file: HF_TOKEN=your_token_here")

# Test Gemini API
if gemini_ok:
    try:
        import google.generativeai as genai
        print("\n🌟 Testing Google Gemini API...")
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content("Hello")
        print(f"✅ Gemini API: WORKING")
        print(f"   Response: {response.text[:100]}")
        
    except Exception as e:
        print(f"❌ Gemini API: FAILED")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   💡 Fix: Check your GEMINI_API_KEY at https://makersuite.google.com/app/apikey")
else:
    print("\n❌ GEMINI_API_KEY not set - skipping test")
    print("   💡 Get key at: https://makersuite.google.com/app/apikey")
    print("   💡 Add to .env file: GEMINI_API_KEY=your_key_here")

# Test Groq API
if groq_ok:
    try:
        from groq import Groq
        print("\n⚡ Testing Groq API...")
        client = Groq(api_key=GROQ_API_KEY)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
        )
        
        result = response.choices[0].message.content
        print(f"✅ Groq API: WORKING")
        print(f"   Response: {result[:100]}")
        
    except Exception as e:
        print(f"❌ Groq API: FAILED")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   💡 Fix: Check your GROQ_API_KEY at https://console.groq.com")
else:
    print("\n❌ GROQ_API_KEY not set - skipping test")
    print("   💡 Get key at: https://console.groq.com")
    print("   💡 Add to .env file: GROQ_API_KEY=your_key_here")

# Test OpenAI API
if openai_ok:
    try:
        from openai import OpenAI
        print("\n🤖 Testing OpenAI API...")
        client = OpenAI(api_key=OPENAI_API_KEY)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=50,
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenAI API: WORKING")
        print(f"   Response: {result[:100]}")
        
    except Exception as e:
        print(f"❌ OpenAI API: FAILED")
        print(f"   Error: {type(e).__name__}: {str(e)}")
        print(f"   💡 Fix: Check your OPENAI_API_KEY at https://platform.openai.com")
else:
    print("\n❌ OPENAI_API_KEY not set - skipping test")
    print("   💡 Get key at: https://platform.openai.com")
    print("   💡 Add to .env file: OPENAI_API_KEY=your_key_here")

print()
print("=" * 70)
print("📝 Summary & Recommendations:")
print("-" * 70)

working = []
if hf_ok: working.append("HF Inference")
if gemini_ok: working.append("Gemini")  
if groq_ok: working.append("Groq")
if openai_ok: working.append("OpenAI")

if not working:
    print("❌ NO API KEYS CONFIGURED!")
    print()
    print("📌 Quick Setup (Choose ONE):")
    print()
    print("1️⃣  HF Inference (FREE, RECOMMENDED for ASEAN):")
    print("   - Visit: https://huggingface.co/settings/tokens")
    print("   - Create a token")
    print("   - Add to .env: HF_TOKEN=your_token_here")
    print()
    print("2️⃣  Google Gemini (FREE, 15 requests/min):")
    print("   - Visit: https://makersuite.google.com/app/apikey")
    print("   - Create an API key")
    print("   - Add to .env: GEMINI_API_KEY=your_key_here")
    print()
    print("3️⃣  Groq (FREE, Ultra-fast):")
    print("   - Visit: https://console.groq.com")
    print("   - Create an API key")
    print("   - Add to .env: GROQ_API_KEY=your_key_here")
else:
    print(f"✅ Working APIs: {', '.join(working)}")
    print()
    print("💡 Make sure your RAG.py uses one of the working providers:")
    print(f"   LLM_PROVIDER = '{working[0].lower().replace(' ', '_')}'")

print("=" * 70)
