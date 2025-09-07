from pydantic import BaseModel, Field
from typing import List

# Pydantic model for the incoming request body.
# This ensures the frontend sends data in the correct format.
class UserProfile(BaseModel):
    disease: str = Field(..., example="I have high blood sugar and feel tired")
    symptoms: List[str] = Field(..., example=["Frequent urination", "Blurry vision"])
    age: int = Field(..., example=45)
    gender: str = Field(..., example="Male")

    class Config:
        # Provides an example for the interactive API docs
        # 'schema_extra' was renamed to 'json_schema_extra' in Pydantic V2
        json_schema_extra = {
            "example": {
                "disease": "Diabetes",
                "symptoms": ["Fatigue", "Increased thirst"],
                "age": 50,
                "gender": "Female"
            }
        }

# Pydantic model for the outgoing response.
# This ensures the API always returns data in a consistent format.
class DietPlanResponse(BaseModel):
    plan: str

