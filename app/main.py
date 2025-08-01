from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.disease_detection import predict_disease
from app.weather_forecast import predict_weather
from app.price_prediction import predict_price
from app.auth import router as auth_router  # ✅ Authentication router

app = FastAPI()

# ✅ Enable CORS so frontend can connect during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Register authentication routes
app.include_router(auth_router)

# ✅ Root endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to AgriSense-AI!"}


# === 🌿 Disease Detection: Upload image for prediction ===
@app.post("/predict-disease", tags=["Disease API"])
async def disease_api(file: UploadFile = File(...)):
    contents = await file.read()
    result = predict_disease(contents)
    return {"prediction": result}


# === 🌦️ Weather Forecast ===
class WeatherRequest(BaseModel):
    location: str

@app.post("/predict-weather", tags=["Weather API"])
async def weather_api(request: WeatherRequest):
    result = predict_weather(request.location)
    return {"forecast": result}


# === 💰 Crop Price Prediction ===
class PriceRequest(BaseModel):
    crop_name: str

@app.post("/predict-price", tags=["Price API"])
async def price_api(request: PriceRequest):
    result = predict_price(request.crop_name)
    return {"price_prediction": result}
