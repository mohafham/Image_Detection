from pydantic import BaseModel, Field
from typing import Optional, Dict


class PredictionRequest(BaseModel):
    """Request model for image prediction"""
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_data": "base64_encoded_image_string"
            }
        }


class PredictionResponse(BaseModel):
    """Response model for image prediction"""
    label: str = Field(..., description="Classification label: 'Real' or 'Fake'")
    confidence: float = Field(..., description="Confidence score (0-100)")
    heatmap_base64: str = Field(..., description="Base64 encoded Grad-CAM heatmap")
    explanation: str = Field(..., description="Natural language explanation from Claude API")
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probability distribution")
    
    class Config:
        json_schema_extra = {
            "example": {
                "label": "Fake",
                "confidence": 95.67,
                "heatmap_base64": "base64_encoded_heatmap",
                "explanation": "The model detected strong AI-generated artifacts...",
                "probabilities": {
                    "Real": 0.043,
                    "Fake": 0.957
                }
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check"""
    status: str = Field(..., description="Health status of the API")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "ok"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid image format",
                "detail": "The uploaded file must be a valid image (JPG, PNG, etc.)"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch image prediction"""
    images: list[str] = Field(..., description="List of base64 encoded images")
    
    class Config:
        json_schema_extra = {
            "example": {
                "images": ["base64_image1", "base64_image2"]
            }
        }


class BatchPredictionResponse(BaseModel):
    """Response model for batch prediction"""
    results: list[PredictionResponse] = Field(..., description="List of prediction results")
    total_processed: int = Field(..., description="Total number of images processed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [],
                "total_processed": 2
            }
        }
