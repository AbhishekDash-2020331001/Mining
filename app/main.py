from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api import routes

app = FastAPI(
    title="Mining Prediction API",
    description="API for mining PPV prediction model training and inference",
    version="1.0.0"
)

# CORS middleware for Android app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your Android app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(routes.router, prefix="/api", tags=["mining"])

@app.get("/")
async def root():
    return {"message": "Mining Prediction API", "status": "running"}

