import torch
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
import io


class ImageClassifier:
    """
    AI-Generated Image Detection Inference Pipeline
    """
    
    def __init__(self, model_path, device=None):
        """
        Initialize the classifier with a trained model
        
        Args:
            model_path (str): Path to saved model weights
            device (str): Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        # Class 0 = AI-GENERATED (FAKE folder in CIFAKE), Class 1 = REAL
        self.labels = ["AI-GENERATED", "REAL"]
        
        # Load model
        self.model = timm.create_model("efficientnet_b3", pretrained=False, num_classes=2)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval().to(self.device)
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess image for inference
        
        Args:
            image: PIL Image or file path
        
        Returns:
            tensor: Preprocessed image tensor
            img_np: Original resized image as numpy array for visualization
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image)).convert("RGB")
        
        img_resized = image.resize((224, 224))
        img_np = np.array(img_resized).astype(np.float32) / 255.0
        
        tensor = self.transform(img_resized).unsqueeze(0).to(self.device)
        
        return tensor, img_np
    
    def predict(self, image):
        """
        Predict if image is real or AI-generated
        
        Args:
            image: PIL Image, file path, or bytes
        
        Returns:
            dict: Prediction results with label, confidence, and probabilities
        """
        tensor, img_np = self.preprocess_image(image)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred = probs.argmax().item()
            confidence = probs[pred].item()
        
        return {
            "label": self.labels[pred],
            "prediction": pred,
            "confidence": confidence,
            "probabilities": {
                "AI-GENERATED": probs[0].item(),
                "REAL": probs[1].item()
            },
            "tensor": tensor,
            "image_np": img_np
        }
    
    def predict_batch(self, images):
        """
        Predict on a batch of images
        
        Args:
            images: List of PIL Images or file paths
        
        Returns:
            list: List of prediction results
        """
        results = []
        for image in images:
            result = self.predict(image)
            results.append(result)
        return results


def load_model(model_path, device=None):
    """
    Helper function to load a trained model
    
    Args:
        model_path (str): Path to saved model weights
        device (str): Device to run inference on
    
    Returns:
        ImageClassifier: Initialized classifier
    """
    return ImageClassifier(model_path, device)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <image_path>")
        print("Example: python inference.py ../models/best_model.pth test_image.jpg")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    
    print("Loading model...")
    classifier = ImageClassifier(model_path)
    
    print(f"Analyzing image: {image_path}")
    result = classifier.predict(image_path)
    
    print("\n" + "=" * 50)
    print(f"Prediction: {result['label']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Probabilities:")
    print(f"  Real: {result['probabilities']['Real']:.2%}")
    print(f"  Fake: {result['probabilities']['Fake']:.2%}")
    print("=" * 50)
