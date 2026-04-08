"""
Verification script to check backend setup
Run this to verify:
1. Model file exists and can be loaded
2. Gemini API is configured
3. All dependencies are installed
"""

import os
import sys

# Fix encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

def check_model_file():
    """Check if model file exists"""
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pth")
    if os.path.exists(model_path):
        print(f"✓ Model file found at: {model_path}")
        return True
    else:
        print(f"✗ Model file NOT found at: {model_path}")
        print("  Please train the model or place best_model.pth in the models/ directory")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'torch',
        'timm',
        'torchvision',
        'fastapi',
        'google.genai',
        'cv2',
        'PIL',
        'numpy'
    ]
    
    missing = []
    for package in required_packages:
        try:
            if package == 'cv2':
                __import__('cv2')
            elif package == 'PIL':
                __import__('PIL')
            elif package == 'google.genai':
                from google import genai
            else:
                __import__(package)
            print(f"✓ {package} installed")
        except ImportError:
            print(f"✗ {package} NOT installed")
            missing.append(package)
    
    return len(missing) == 0, missing

def check_gemini_api():
    """Check if Gemini API key is configured"""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key and api_key != "your_google_ai_studio_api_key_here":
        print(f"✓ GOOGLE_API_KEY is configured")
        return True
    else:
        print(f"✗ GOOGLE_API_KEY is NOT configured")
        print("  Please set GOOGLE_API_KEY in your .env file")
        return False

def test_model_loading():
    """Test if model can be loaded"""
    try:
        import torch
        import timm
        
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "best_model.pth")
        
        if not os.path.exists(model_path):
            print("⚠ Skipping model loading test (model file not found)")
            return True
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        print(f"✓ Model loaded successfully on {device}")
        return True
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        from dotenv import load_dotenv
        from google import genai
        
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        
        if not api_key or api_key == "your_google_ai_studio_api_key_here":
            print("⚠ Skipping Gemini test (API key not configured)")
            return True
        
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents="Hello, this is a test."
        )
        
        print(f"✓ Gemini API connection successful")
        print(f"  Response: {response.text[:50]}...")
        return True
    except Exception as e:
        print(f"✗ Gemini API test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Backend Verification Script")
    print("=" * 60)
    print()
    
    results = []
    
    print("1. Checking dependencies...")
    deps_ok, missing = check_dependencies()
    results.append(deps_ok)
    if not deps_ok:
        print(f"\n  Install missing packages with:")
        print(f"  pip install {' '.join(missing)}")
    print()
    
    print("2. Checking model file...")
    model_ok = check_model_file()
    results.append(model_ok)
    print()
    
    print("3. Checking Gemini API configuration...")
    api_ok = check_gemini_api()
    results.append(api_ok)
    print()
    
    if deps_ok:
        print("4. Testing model loading...")
        load_ok = test_model_loading()
        results.append(load_ok)
        print()
        
        print("5. Testing Gemini API connection...")
        gemini_ok = test_gemini_connection()
        results.append(gemini_ok)
        print()
    
    print("=" * 60)
    if all(results):
        print("✓ All checks passed! Backend is ready.")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
