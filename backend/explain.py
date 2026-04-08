from google import genai
import os

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

SYSTEM_PROMPT = """
You are an expert in digital forensics and AI-generated image detection.
Given a classification result and confidence score for an image, generate
a clear, concise 2-3 sentence explanation of likely artifacts or features
that indicate whether the image is real or AI-generated.
Focus on: texture anomalies, lighting inconsistencies, facial/hand distortions,
background incoherence, and unnatural symmetry.
Be specific and technical but understandable.
"""


def generate_explanation(label: str, confidence: float, top_regions: str) -> str:
    """
    Generate natural language explanation for image classification using Gemini API
    
    Args:
        label (str): Classification label ('Real' or 'Fake')
        confidence (float): Confidence score (0-1)
        top_regions (str): Description of highlighted regions from Grad-CAM
    
    Returns:
        str: Natural language explanation
    """
    prompt = f"""
    {SYSTEM_PROMPT}
    
    Classification: {label}
    Confidence: {confidence:.1%}
    Highlighted regions from Grad-CAM: {top_regions}

    Explain why this image is classified as {label}.
    """
    
    try:
        if not client:
            return generate_fallback_explanation(label, confidence)
        
        # Generate content using new google.genai API
        response = client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=prompt
        )
        
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        # Fallback explanation if API fails
        return generate_fallback_explanation(label, confidence)


def generate_fallback_explanation(label: str, confidence: float) -> str:
    """
    Generate a fallback explanation if Gemini API is unavailable
    
    Args:
        label (str): Classification label
        confidence (float): Confidence score (0-1)
    
    Returns:
        str: Fallback explanation
    """
    if label == "AI-GENERATED":
        if confidence > 0.9:
            return "The model detected strong AI-generated artifacts in this image with very high confidence. Common indicators include unnatural texture patterns, inconsistent lighting, or subtle geometric anomalies typical of generative models."
        elif confidence > 0.7:
            return "The image shows notable characteristics of AI-generated content. The model identified patterns consistent with synthetic image generation, such as unusual texture smoothness or minor inconsistencies in object boundaries."
        else:
            return "The model leans toward classifying this as AI-generated, though with moderate confidence. There may be subtle artifacts present, but they are not definitive enough for a strong determination."
    else:  # Real
        if confidence > 0.9:
            return "The image exhibits authentic photographic characteristics with very high confidence. Natural noise patterns, consistent lighting, and realistic texture variation indicate this is a genuine photograph rather than AI-generated content."
        elif confidence > 0.7:
            return "The model identifies this as a real photograph with notable confidence. Authentic imaging artifacts, natural imperfections, and consistent physical properties suggest genuine capture rather than synthetic generation."
        else:
            return "The image appears more likely to be real photography, though with moderate confidence. Most characteristics align with authentic capture, but some ambiguous features prevent a stronger determination."


def generate_batch_explanations(results):
    """
    Generate explanations for a batch of classification results
    
    Args:
        results (list): List of dicts with 'label', 'confidence', and 'top_regions'
    
    Returns:
        list: List of explanations
    """
    explanations = []
    for result in results:
        explanation = generate_explanation(
            result['label'],
            result['confidence'],
            result.get('top_regions', 'various regions')
        )
        explanations.append(explanation)
    return explanations


if __name__ == "__main__":
    # Test the explanation generator
    print("Testing explanation generator...\n")
    
    # Test case 1: High confidence Fake
    print("Test 1: High confidence AI-generated")
    explanation = generate_explanation("Fake", 0.95, "face and hands regions")
    print(f"Explanation: {explanation}\n")
    
    # Test case 2: High confidence Real
    print("Test 2: High confidence Real")
    explanation = generate_explanation("Real", 0.92, "background and lighting")
    print(f"Explanation: {explanation}\n")
    
    # Test case 3: Low confidence
    print("Test 3: Low confidence")
    explanation = generate_explanation("Fake", 0.65, "texture patterns")
    print(f"Explanation: {explanation}")
