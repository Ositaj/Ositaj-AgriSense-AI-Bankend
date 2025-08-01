# app/price_predictor.py

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

# Simulated dynamic crop price logic
def predict_price(crop_name: str) -> str:
    crop_prices = {
        "rice": 1200,
        "maize": 800,
        "cassava": 600,
        "yam": 1000,
        "tomato": 900,
        "onion": 750,
        "soybean": 950
    }
    crop = crop_name.strip().lower()
    price = crop_prices.get(crop, 500)
    return f"Predicted price for {crop_name.capitalize()}: ₦{price} per kg"

class CropRequest(BaseModel):
    crop: str

@router.post("/predict-price")
def predict_price_endpoint(request: CropRequest):
    return {"prediction": predict_price(request.crop)}
