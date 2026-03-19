"""
LLM API Diagnostics
===================
This script helps diagnose issues with your LLM API configuration.
Run this to check if your API keys and endpoints are working.
"""

import os
from dotenv import load_dotenv
load_dotenv()

def check_env_vars():
    """Check if environment variables are set"""
    print("\n" + "="*70)
    print("🔍 CHECKING ENVIRONMENT VARIABLES")
    print("="*70)
    
    env_vars = {
        'HF_TOKEN': os.getenv('HF_TOKEN'),
        'GEMINI_API_KEY': os.getenv('GEMINI_API_KEY'),
        'GROQ_API_KEY': os.getenv('GROQ_API_KEY'),
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
    }
    
    for key, value in env_vars.items():
        if value and value != f'YOUR_{key}':
            print(f"✓ {key}: Set ({value[:10]}...)")
        else:
            print(f"✗ {key}: NOT SET or using placeholder")
    
    return env_vars

def test_hf_inference():
    """Test Hugging Face Inference API"""
    print("\n" + "="*70)
    print("🤗 TESTING HUGGING FACE INFERENCE API")
    print("="*70)
    
    try:
        from huggingface_hub import InferenceClient
        
        token = os.getenv('HF_TOKEN')
        if not token or token == 'YOUR_HF_TOKEN':
            print("✗ HF_TOKEN not configured")
            print("💡 Get your token at: https://huggingface.co/settings/tokens")
            return False
        
        print(f"✓ Token found: {token[:15]}...")
        
        # Test with a simple model
        print("📡 Testing connection...")
        client = InferenceClient(token=token)
        
        # Try a simple text generation request
        model = "aisingapore/llama3-8b-cpt-sea-lionv2.1-instruct"
        print(f"📝 Testing model: {model}")
        
        response = client.chat_completion(
            messages=[{"role": "user", "content": "Hello, reply with just 'OK'"}],
            model=model,
            max_tokens=10,
        )
        
        answer = response.choices[0].message.content
        print(f"✓ API Response: '{answer}'")
        print("✅ Hugging Face Inference API is WORKING!")
        return True
        
    except ImportError:
        print("✗ huggingface_hub not installed")
        print("💡 Install: pip install huggingface_hub")
        return False
    except Exception as e:
        print(f"✗ API call failed: {type(e).__name__}: {str(e)}")
        
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("💡 Fix: Your HF_TOKEN might be invalid. Generate a new one at:")
            print("   https://huggingface.co/settings/tokens")
            print("   Make sure to select 'Read' permissions")
        elif "model" in str(e).lower():
            print("💡 Fix: The model might not support Inference API")
            print("   Try using: 'microsoft/Phi-3-mini-4k-instruct' instead")
        elif "rate" in str(e).lower():
            print("💡 Fix: Rate limit exceeded. Wait a few minutes and try again")
        
        return False

def test_gemini():
    """Test Google Gemini API"""
    print("\n" + "="*70)
    print("🌟 TESTING GOOGLE GEMINI API")
    print("="*70)
    
    try:
        import google.generativeai as genai
        
        api_key = os.getenv('GEMINI_API_KEY')
        if not api_key or api_key == 'YOUR_GEMINI_API_KEY':
            print("✗ GEMINI_API_KEY not configured")
            print("💡 Get your key at: https://makersuite.google.com/app/apikey")
            return False
        
        print(f"✓ API Key found: {api_key[:15]}...")
        
        print("📡 Testing connection...")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content("Hello, reply with just 'OK'")
        answer = response.text
        
        print(f"✓ API Response: '{answer}'")
        print("✅ Google Gemini API is WORKING!")
        return True
        
    except ImportError:
        print("✗ google-generativeai not installed")
        print("💡 Install: pip install google-generativeai")
        return False
    except Exception as e:
        print(f"✗ API call failed: {type(e).__name__}: {str(e)}")
        
        if "API_KEY" in str(e).upper() or "401" in str(e):
            print("💡 Fix: Your GEMINI_API_KEY might be invalid. Generate a new one at:")
            print("   https://makersuite.google.com/app/apikey")
        
        return False

def test_groq():
    """Test Groq API"""
    print("\n" + "="*70)
    print("⚡ TESTING GROQ API")
    print("="*70)
    
    try:
        from groq import Groq
        
        api_key = os.getenv('GROQ_API_KEY')
        if not api_key or api_key == 'YOUR_GROQ_API_KEY':
            print("✗ GROQ_API_KEY not configured")
            print("💡 Get your key at: https://console.groq.com")
            return False
        
        print(f"✓ API Key found: {api_key[:15]}...")
        
        print("📡 Testing connection...")
        client = Groq(api_key=api_key)
        
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "Hello, reply with just 'OK'"}],
            max_tokens=10,
        )
        
        answer = response.choices[0].message.content
        print(f"✓ API Response: '{answer}'")
        print("✅ Groq API is WORKING!")
        return True
        
    except ImportError:
        print("✗ groq not installed")
        print("💡 Install: pip install groq")
        return False
    except Exception as e:
        print(f"✗ API call failed: {type(e).__name__}: {str(e)}")
        
        if "api_key" in str(e).lower() or "401" in str(e):
            print("💡 Fix: Your GROQ_API_KEY might be invalid. Generate a new one at:")
            print("   https://console.groq.com")
        
        return False

def main():
    """Run all diagnostics"""
    print("\n🔬 LLM API DIAGNOSTICS")
    print("="*70)
    
    # Check environment variables
    env_vars = check_env_vars()
    
    # Test APIs
    results = {
        'hf_inference': test_hf_inference(),
        'gemini': test_gemini(),
        'groq': test_groq(),
    }
    
    # Summary
    print("\n" + "="*70)
    print("📊 SUMMARY")
    print("="*70)
    
    working_providers = [k for k, v in results.items() if v]
    
    if working_providers:
        print(f"✅ Working providers: {', '.join(working_providers)}")
        print(f"\n💡 Recommendation: Update LLM_PROVIDER in RAG.py to use: '{working_providers[0]}'")
    else:
        print("✗ No working providers found")
        print("\n💡 Next steps:")
        print("1. Get a FREE API key from one of these providers:")
        print("   - Hugging Face: https://huggingface.co/settings/tokens")
        print("   - Google Gemini: https://makersuite.google.com/app/apikey")
        print("   - Groq: https://console.groq.com")
        print("2. Add it to your .env file")
        print("3. Run this script again to verify")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
